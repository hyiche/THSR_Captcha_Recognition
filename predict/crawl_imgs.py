import cv2
import time
import tensorflow as tf
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from scipy.special import softmax
from random import randint, sample
from requests import Session, adapters
from datetime import date, datetime, timedelta
from argparse import ArgumentParser, Namespace
from tensorflow.keras import Model as tf_Model
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import newaxis, ndarray, array, where, column_stack, argmax


class AllDigitsAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='all_digits_acc', **kwargs):
        super(AllDigitsAccuracy, self).__init__(name=name, **kwargs)
        self.ad_acc = self.add_weight(name='adacc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # from one-hot encoding to sparse classes type (tensor: 3d ---> 2d)
        y_true_ = tf.argmax(y_true, axis=2)

        # because model output didn't pass softmax operation
        y_pred_ = tf.keras.activations.softmax(y_pred)

        # y_pred_ becomes sparse classes type(tensor: 3d ---> 2d)
        y_pred_ = tf.argmax(y_pred_, axis=2)

        # compare y_true_ and y_pred_, output is tf.bool(tensor: 2d ---> 2d)
        v = y_true_ == y_pred_

        # check whether every digit is correct (tensor: 2d ---> 1d)
        v = tf.reduce_all(v, axis=1)

        # cast to float for next progress to compute reduce_mean
        v = tf.cast(v, dtype=tf.dtypes.float32)
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, self.dtype)
        #     sample_weight = tf.broadcast_to(sample_weight, values.shape)
        #     values = tf.multiply(values, sample_weight)
        self.ad_acc.assign_add(tf.reduce_sum(v))

    def result(self):
        return self.ad_acc

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.ad_acc.assign(0.0)


def parse_args() -> Namespace:
    parser = ArgumentParser("parameters configuration")
    # HTTPConfig
    parser.add_argument('--BASE_URL', type=str, default='https://irs.thsrc.com.tw')
    parser.add_argument('--BOOKING_PAGE_URL', type=str, default='https://irs.thsrc.com.tw/IMINT/?locale=tw')
    parser.add_argument('--CONFIRM_TRAIN_URL', type=str,
                        default='https://irs.thsrc.com.tw/IMINT/?wicket:interface=:1:BookingS2Form::IFormSubmitListener')
    parser.add_argument('--CONFIRM_TICKET_URL', type=str,
                        default='https://irs.thsrc.com.tw/IMINT/?wicket:interface=:2:BookingS3Form::IFormSubmitListener')
    parser.add_argument('--SUBMIT_FORM_URL', type=str, default='https://irs.thsrc.com.tw/IMINT/;jsessionid={}?wicket:interface=:0:BookingS1Form::IFormSubmitListener')

    # HTTPHeader
    parser.add_argument('--ACCEPT_IMG', type=str, default='image/webp,*/*')
    parser.add_argument('--ACCEPT_ENCODING', type=str, default='gzip, deflate, br')
    parser.add_argument('--ACCEPT_LANGUAGE', type=str, default='zh-TW,zh;q=0.8,en-US;q=0.5,en;q=0.3')
    parser.add_argument('--ACCEPT_HTML', type=str,
                        default='text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
    parser.add_argument('--USER_AGENT', type=str,
                        default='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0')
    # Host URL
    parser.add_argument('--BOOKING_PAGE_HOST', type=str, default='irs.thsrc.com.tw')
    return parser.parse_args(args=[])


def remove_curve(img_arr: ndarray) -> ndarray:
    res = img_arr.copy()
    height, width = res.shape
    img_arr[:, 5: width-5] = 0
    image_idx = where(img_arr == 255)
    xx, yy = array([image_idx[1]]), height - image_idx[0]
    poly_reg, l_reg = PolynomialFeatures(degree=2), LinearRegression(n_jobs=-1)
    xx_poly = poly_reg.fit_transform(xx.T)
    l_reg.fit(xx_poly, yy)
    xx2 = array([[i for i in range(width)]])
    xx2_poly = poly_reg.fit_transform(xx2.T)
    for ele in column_stack([l_reg.predict(xx2_poly).round(0), xx2[0]]):
        # print(ele)
        loc = height - int(ele[0])
        # if newimg[loc-4:loc+4,int(ele[1])] == 255:
        # newimg[loc-3:loc+3,int(ele[1])] = 0  # this line can remove curve。
        res[loc - 3: loc + 3, int(ele[1])] = 255 - res[loc - 3: loc + 3, int(ele[1])]  # 弧線過處，黑白互換。
    return res


def test_predict(test_img: ndarray, classifier: tf_Model) -> str:
    d_ = {0: '2', 1: '3', 2: '4', 3: '5', 4: '7', 5: '9', 6: 'A', 7: 'C', 8: 'F', 9: 'H',
          10: 'K', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'T', 17: 'Y', 18: 'Z'}
    temp = cv2.fastNlMeansDenoisingColored(test_img, None, 30, 30, 7, 31)
    ret, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY_INV)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = remove_curve(img_arr=temp)
    test_captcha_ = cv2.resize(temp, (140, 48), interpolation=cv2.INTER_CUBIC)
    test_captcha_ = test_captcha_[newaxis, :]  # to feed in model.
    pred = classifier.predict(x=test_captcha_, batch_size=1, verbose=0)
    pred = softmax(pred[0], axis=1)
    pred = argmax(pred, axis=1)
    pred = [d_[i_pred] for i_pred in pred]
    return ''.join(pred)


def parse_security_img_url(arg: Namespace, html: bytes) -> str:
    booking_page_dic = {"security_code_img": {"id": "BookingS1Form_homeCaptcha_passCode"}}  # parse html element
    page = BeautifulSoup(html, features="html.parser")
    element = page.find(**booking_page_dic["security_code_img"])
    return arg.BASE_URL + element["src"]


def pass_first_page_with_captcha(arg: Namespace, number: int = 100):
    reconstructed_model = load_model('./saved_model/thsr_model', custom_objects={'AllDigitsAccuracy': AllDigitsAccuracy})
    # 欲送出的表格欄位資訊
    forms = {'BookingS1Form:hf:0': '', 'selectStartStation': 6, 'selectDestinationStation': 3,
             'trainCon:trainRadioGroup': 0, 'seatCon:seatRadioGroup': 'radio17',
             'bookingMethod': 0, 'toTimeInputField': '2020/12/30', 'toTimeTable': '300P',
             'toTrainIDInputField': 0, 'backTimeInputField': '', 'backTimeTable': '',
             'backTrainIDInputField': '', 'ticketPanel:rows:0:ticketAmount': '1F',
             'ticketPanel:rows:1:ticketAmount': '0H', 'ticketPanel:rows:2:ticketAmount': '0W',
             'ticketPanel:rows:3:ticketAmount': '0E', 'ticketPanel:rows:4:ticketAmount': '0P',
             'homeCaptcha:securityCode': 'Z24N'}
    count, wrong_idx_list = 0, []
    for i in range(1, number+1):
        sess = Session()
        sess.mount("https://", adapters.HTTPAdapter(max_retries=3))
        common_head_html: dict = {"Host": arg.BOOKING_PAGE_HOST,
                                  "User-Agent": arg.USER_AGENT,
                                  "Accept": arg.ACCEPT_HTML,
                                  "Accept-Language": arg.ACCEPT_LANGUAGE,
                                  "Accept-Encoding": arg.ACCEPT_ENCODING}

        # 獲得驗證碼圖
        book_page = sess.get(url=arg.BOOKING_PAGE_URL, headers=common_head_html, allow_redirects=True)
        img_url = parse_security_img_url(arg=arg, html=book_page.content)
        img_response = sess.get(url=img_url, headers=common_head_html)
        image = Image.open(BytesIO(img_response.content))
        img_array = array(image)

        # 將驗證碼圖讓模型預測，預測結果放入forms
        s = test_predict(test_img=img_array, classifier=reconstructed_model)
        forms['homeCaptcha:securityCode'] = s

        # 隨機決定訂票時間(目前表格只填去程)
        date_now = date.today()
        date_ = date_now + timedelta(days=randint(a=0, b=27))  # 規定28天內開放訂票
        t = date_.strftime("%Y/%m/%d")
        forms['toTimeInputField'] = t

        # 隨機選擇起訖站
        stations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        start_station, dest_station = sample(population=stations, k=2)
        forms['selectStartStation'] = start_station
        forms['selectDestinationStation'] = dest_station

        # 送出表單給server，看是否通過驗證碼(即通過訂票網頁第一頁)
        url = arg.SUBMIT_FORM_URL.format(sess.cookies["JSESSIONID"])
        resp = sess.post(url=url, headers=common_head_html, params=forms, allow_redirects=True)
        soup = BeautifulSoup(resp.text, "lxml")
        find_err = soup.find_all("", {"class": "feedbackPanelERROR"})
        right_imgs_path, wrong_imgs_path = './temp_right/', './temp_wrong/'
        if len(find_err):   # wrong
            wrong_idx_list.append(i)
            print('i: {:4d}, prediction: {}       (X)'.format(i, s))
            cv2.imwrite(wrong_imgs_path + s + '.jpg', img_array)
        else:               # right
            count += 1
            print('i: {:4d}, prediction: {}   (O)'.format(i, s))
            cv2.imwrite(right_imgs_path + s + '.jpg', img_array)
    # number=1時，下兩行資訊不重要，不需print出。
    print('Accuracy:({}/{})'.format(count, number))
    print('wrong_idx_list:\n', wrong_idx_list)


# 每n秒執行一次pass_first_page_with_captcha()
def crawl_captcha_per_n_secs(n):
    while True:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pass_first_page_with_captcha(arg=args, number=1)  # 寫死每次執行只爬取一張圖。
        time.sleep(n)


if __name__ == '__main__':
    args = parse_args()
    # crawl_captcha_per_n_secs(n=8)  # 設定每n秒爬取一次。
    number = int(input('欲爬取多少張驗證碼? ---> '))
    pass_first_page_with_captcha(arg=args, number=number)
