??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:@*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:@*
dtype0
j
	bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	bn1/gamma
c
bn1/gamma/Read/ReadVariableOpReadVariableOp	bn1/gamma*
_output_shapes
:@*
dtype0
h
bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bn1/beta
a
bn1/beta/Read/ReadVariableOpReadVariableOpbn1/beta*
_output_shapes
:@*
dtype0
v
bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namebn1/moving_mean
o
#bn1/moving_mean/Read/ReadVariableOpReadVariableOpbn1/moving_mean*
_output_shapes
:@*
dtype0
~
bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namebn1/moving_variance
w
'bn1/moving_variance/Read/ReadVariableOpReadVariableOpbn1/moving_variance*
_output_shapes
:@*
dtype0
}
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameconv2/kernel
v
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*'
_output_shapes
:@?*
dtype0
m

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv2/bias
f
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes	
:?*
dtype0
k
	bn2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn2/gamma
d
bn2/gamma/Read/ReadVariableOpReadVariableOp	bn2/gamma*
_output_shapes	
:?*
dtype0
i
bn2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn2/beta
b
bn2/beta/Read/ReadVariableOpReadVariableOpbn2/beta*
_output_shapes	
:?*
dtype0
w
bn2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namebn2/moving_mean
p
#bn2/moving_mean/Read/ReadVariableOpReadVariableOpbn2/moving_mean*
_output_shapes	
:?*
dtype0

bn2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namebn2/moving_variance
x
'bn2/moving_variance/Read/ReadVariableOpReadVariableOpbn2/moving_variance*
_output_shapes	
:?*
dtype0
y
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_namedense1/kernel
r
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*!
_output_shapes
:???*
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:?*
dtype0
w
digit1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedigit1/kernel
p
!digit1/kernel/Read/ReadVariableOpReadVariableOpdigit1/kernel*
_output_shapes
:	?*
dtype0
n
digit1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedigit1/bias
g
digit1/bias/Read/ReadVariableOpReadVariableOpdigit1/bias*
_output_shapes
:*
dtype0
w
digit2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedigit2/kernel
p
!digit2/kernel/Read/ReadVariableOpReadVariableOpdigit2/kernel*
_output_shapes
:	?*
dtype0
n
digit2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedigit2/bias
g
digit2/bias/Read/ReadVariableOpReadVariableOpdigit2/bias*
_output_shapes
:*
dtype0
w
digit3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedigit3/kernel
p
!digit3/kernel/Read/ReadVariableOpReadVariableOpdigit3/kernel*
_output_shapes
:	?*
dtype0
n
digit3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedigit3/bias
g
digit3/bias/Read/ReadVariableOpReadVariableOpdigit3/bias*
_output_shapes
:*
dtype0
w
digit4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedigit4/kernel
p
!digit4/kernel/Read/ReadVariableOpReadVariableOpdigit4/kernel*
_output_shapes
:	?*
dtype0
n
digit4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedigit4/bias
g
digit4/bias/Read/ReadVariableOpReadVariableOpdigit4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
^
adaccVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameadacc
W
adacc/Read/ReadVariableOpReadVariableOpadacc*
_output_shapes
: *
dtype0
?
Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/conv1/kernel/m
?
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*&
_output_shapes
:@*
dtype0
z
Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/conv1/bias/m
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
_output_shapes
:@*
dtype0
x
Adam/bn1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/bn1/gamma/m
q
$Adam/bn1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn1/gamma/m*
_output_shapes
:@*
dtype0
v
Adam/bn1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameAdam/bn1/beta/m
o
#Adam/bn1/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn1/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameAdam/conv2/kernel/m
?
'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*'
_output_shapes
:@?*
dtype0
{
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/conv2/bias/m
t
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes	
:?*
dtype0
y
Adam/bn2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/bn2/gamma/m
r
$Adam/bn2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn2/gamma/m*
_output_shapes	
:?*
dtype0
w
Adam/bn2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameAdam/bn2/beta/m
p
#Adam/bn2/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn2/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*%
shared_nameAdam/dense1/kernel/m
?
(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m*!
_output_shapes
:???*
dtype0
}
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/dense1/bias/m
v
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/digit1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit1/kernel/m
~
(Adam/digit1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/digit1/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/digit1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit1/bias/m
u
&Adam/digit1/bias/m/Read/ReadVariableOpReadVariableOpAdam/digit1/bias/m*
_output_shapes
:*
dtype0
?
Adam/digit2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit2/kernel/m
~
(Adam/digit2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/digit2/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/digit2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit2/bias/m
u
&Adam/digit2/bias/m/Read/ReadVariableOpReadVariableOpAdam/digit2/bias/m*
_output_shapes
:*
dtype0
?
Adam/digit3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit3/kernel/m
~
(Adam/digit3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/digit3/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/digit3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit3/bias/m
u
&Adam/digit3/bias/m/Read/ReadVariableOpReadVariableOpAdam/digit3/bias/m*
_output_shapes
:*
dtype0
?
Adam/digit4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit4/kernel/m
~
(Adam/digit4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/digit4/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/digit4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit4/bias/m
u
&Adam/digit4/bias/m/Read/ReadVariableOpReadVariableOpAdam/digit4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/conv1/kernel/v
?
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
:@*
dtype0
z
Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes
:@*
dtype0
x
Adam/bn1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/bn1/gamma/v
q
$Adam/bn1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn1/gamma/v*
_output_shapes
:@*
dtype0
v
Adam/bn1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameAdam/bn1/beta/v
o
#Adam/bn1/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn1/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameAdam/conv2/kernel/v
?
'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*'
_output_shapes
:@?*
dtype0
{
Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/conv2/bias/v
t
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
_output_shapes	
:?*
dtype0
y
Adam/bn2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/bn2/gamma/v
r
$Adam/bn2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn2/gamma/v*
_output_shapes	
:?*
dtype0
w
Adam/bn2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameAdam/bn2/beta/v
p
#Adam/bn2/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn2/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*%
shared_nameAdam/dense1/kernel/v
?
(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v*!
_output_shapes
:???*
dtype0
}
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/dense1/bias/v
v
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/digit1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit1/kernel/v
~
(Adam/digit1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/digit1/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/digit1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit1/bias/v
u
&Adam/digit1/bias/v/Read/ReadVariableOpReadVariableOpAdam/digit1/bias/v*
_output_shapes
:*
dtype0
?
Adam/digit2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit2/kernel/v
~
(Adam/digit2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/digit2/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/digit2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit2/bias/v
u
&Adam/digit2/bias/v/Read/ReadVariableOpReadVariableOpAdam/digit2/bias/v*
_output_shapes
:*
dtype0
?
Adam/digit3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit3/kernel/v
~
(Adam/digit3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/digit3/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/digit3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit3/bias/v
u
&Adam/digit3/bias/v/Read/ReadVariableOpReadVariableOpAdam/digit3/bias/v*
_output_shapes
:*
dtype0
?
Adam/digit4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/digit4/kernel/v
~
(Adam/digit4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/digit4/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/digit4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/digit4/bias/v
u
&Adam/digit4/bias/v/Read/ReadVariableOpReadVariableOpAdam/digit4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?x
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?x
value?xB?x B?x
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
	optimizer
_layers
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
?
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
?
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>trainable_variables
?regularization_losses
@	variables
A	keras_api
R
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
R
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
R
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
h

Tkernel
Ubias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
h

Zkernel
[bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
h

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
h

fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
R
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
R
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
R
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
R
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
R
|trainable_variables
}regularization_losses
~	variables
	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?'m?(m?3m?4m?:m?;m?Jm?Km?Tm?Um?Zm?[m?`m?am?fm?gm? v?!v?'v?(v?3v?4v?:v?;v?Jv?Kv?Tv?Uv?Zv?[v?`v?av?fv?gv?
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
?
 0
!1
'2
(3
34
45
:6
;7
J8
K9
T10
U11
Z12
[13
`14
a15
f16
g17
 
?
 0
!1
'2
(3
)4
*5
36
47
:8
;9
<10
=11
J12
K13
T14
U15
Z16
[17
`18
a19
f20
g21
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
trainable_variables
?layer_metrics
regularization_losses
	variables
?metrics
 
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
trainable_variables
?layer_metrics
regularization_losses
	variables
?metrics
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
"trainable_variables
?layer_metrics
#regularization_losses
$	variables
?metrics
 
TR
VARIABLE_VALUE	bn1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEbn1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbn1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
)2
*3
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
+trainable_variables
?layer_metrics
,regularization_losses
-	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
/trainable_variables
?layer_metrics
0regularization_losses
1	variables
?metrics
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
5trainable_variables
?layer_metrics
6regularization_losses
7	variables
?metrics
 
TR
VARIABLE_VALUE	bn2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEbn2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbn2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
<2
=3
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
>trainable_variables
?layer_metrics
?regularization_losses
@	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Btrainable_variables
?layer_metrics
Cregularization_losses
D	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Ftrainable_variables
?layer_metrics
Gregularization_losses
H	variables
?metrics
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Ltrainable_variables
?layer_metrics
Mregularization_losses
N	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Ptrainable_variables
?layer_metrics
Qregularization_losses
R	variables
?metrics
YW
VARIABLE_VALUEdigit1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdigit1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Vtrainable_variables
?layer_metrics
Wregularization_losses
X	variables
?metrics
YW
VARIABLE_VALUEdigit2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdigit2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1
 

Z0
[1
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
\trainable_variables
?layer_metrics
]regularization_losses
^	variables
?metrics
YW
VARIABLE_VALUEdigit3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdigit3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
 

`0
a1
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
btrainable_variables
?layer_metrics
cregularization_losses
d	variables
?metrics
YW
VARIABLE_VALUEdigit4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdigit4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
htrainable_variables
?layer_metrics
iregularization_losses
j	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
ltrainable_variables
?layer_metrics
mregularization_losses
n	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
ptrainable_variables
?layer_metrics
qregularization_losses
r	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
ttrainable_variables
?layer_metrics
uregularization_losses
v	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
xtrainable_variables
?layer_metrics
yregularization_losses
z	variables
?metrics
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
|trainable_variables
?layer_metrics
}regularization_losses
~	variables
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
<2
=3
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 

?0
?1
?2
 
 
 
 
 
 
 
 
 
 

)0
*1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

<0
=1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
9

?adacc
?ad_acc
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
OM
VARIABLE_VALUEadacc4keras_api/metrics/2/adacc/.ATTRIBUTES/VARIABLE_VALUE

?0

?	variables
{y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/digit4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/digit4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputPlaceholder*,
_output_shapes
:?????????0?*
dtype0*!
shape:?????????0?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv1/kernel
conv1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2/kernel
conv2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_variancedense1/kerneldense1/biasdigit4/kerneldigit4/biasdigit3/kerneldigit3/biasdigit2/kerneldigit2/biasdigit1/kerneldigit1/bias*"
Tin
2*
Tout
2*+
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_238652
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOpbn1/gamma/Read/ReadVariableOpbn1/beta/Read/ReadVariableOp#bn1/moving_mean/Read/ReadVariableOp'bn1/moving_variance/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOpbn2/gamma/Read/ReadVariableOpbn2/beta/Read/ReadVariableOp#bn2/moving_mean/Read/ReadVariableOp'bn2/moving_variance/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!digit1/kernel/Read/ReadVariableOpdigit1/bias/Read/ReadVariableOp!digit2/kernel/Read/ReadVariableOpdigit2/bias/Read/ReadVariableOp!digit3/kernel/Read/ReadVariableOpdigit3/bias/Read/ReadVariableOp!digit4/kernel/Read/ReadVariableOpdigit4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpadacc/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp$Adam/bn1/gamma/m/Read/ReadVariableOp#Adam/bn1/beta/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp$Adam/bn2/gamma/m/Read/ReadVariableOp#Adam/bn2/beta/m/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp(Adam/digit1/kernel/m/Read/ReadVariableOp&Adam/digit1/bias/m/Read/ReadVariableOp(Adam/digit2/kernel/m/Read/ReadVariableOp&Adam/digit2/bias/m/Read/ReadVariableOp(Adam/digit3/kernel/m/Read/ReadVariableOp&Adam/digit3/bias/m/Read/ReadVariableOp(Adam/digit4/kernel/m/Read/ReadVariableOp&Adam/digit4/bias/m/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp$Adam/bn1/gamma/v/Read/ReadVariableOp#Adam/bn1/beta/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp$Adam/bn2/gamma/v/Read/ReadVariableOp#Adam/bn2/beta/v/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp(Adam/digit1/kernel/v/Read/ReadVariableOp&Adam/digit1/bias/v/Read/ReadVariableOp(Adam/digit2/kernel/v/Read/ReadVariableOp&Adam/digit2/bias/v/Read/ReadVariableOp(Adam/digit3/kernel/v/Read/ReadVariableOp&Adam/digit3/bias/v/Read/ReadVariableOp(Adam/digit4/kernel/v/Read/ReadVariableOp&Adam/digit4/bias/v/Read/ReadVariableOpConst*Q
TinJ
H2F	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_239837
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/bias	bn1/gammabn1/betabn1/moving_meanbn1/moving_varianceconv2/kernel
conv2/bias	bn2/gammabn2/betabn2/moving_meanbn2/moving_variancedense1/kerneldense1/biasdigit1/kerneldigit1/biasdigit2/kerneldigit2/biasdigit3/kerneldigit3/biasdigit4/kerneldigit4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1adaccAdam/conv1/kernel/mAdam/conv1/bias/mAdam/bn1/gamma/mAdam/bn1/beta/mAdam/conv2/kernel/mAdam/conv2/bias/mAdam/bn2/gamma/mAdam/bn2/beta/mAdam/dense1/kernel/mAdam/dense1/bias/mAdam/digit1/kernel/mAdam/digit1/bias/mAdam/digit2/kernel/mAdam/digit2/bias/mAdam/digit3/kernel/mAdam/digit3/bias/mAdam/digit4/kernel/mAdam/digit4/bias/mAdam/conv1/kernel/vAdam/conv1/bias/vAdam/bn1/gamma/vAdam/bn1/beta/vAdam/conv2/kernel/vAdam/conv2/bias/vAdam/bn2/gamma/vAdam/bn2/beta/vAdam/dense1/kernel/vAdam/dense1/bias/vAdam/digit1/kernel/vAdam/digit1/bias/vAdam/digit2/kernel/vAdam/digit2/bias/vAdam/digit3/kernel/vAdam/digit3/bias/vAdam/digit4/kernel/vAdam/digit4/bias/v*P
TinI
G2E*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_240053??
?
?
B__inference_digit3_layer_call_and_return_conditional_losses_238096

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
K
/__inference_reshape_digit1_layer_call_fn_239535

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_2381772
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ň
?
__inference__traced_save_239837
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop(
$savev2_bn1_gamma_read_readvariableop'
#savev2_bn1_beta_read_readvariableop.
*savev2_bn1_moving_mean_read_readvariableop2
.savev2_bn1_moving_variance_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop(
$savev2_bn2_gamma_read_readvariableop'
#savev2_bn2_beta_read_readvariableop.
*savev2_bn2_moving_mean_read_readvariableop2
.savev2_bn2_moving_variance_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_digit1_kernel_read_readvariableop*
&savev2_digit1_bias_read_readvariableop,
(savev2_digit2_kernel_read_readvariableop*
&savev2_digit2_bias_read_readvariableop,
(savev2_digit3_kernel_read_readvariableop*
&savev2_digit3_bias_read_readvariableop,
(savev2_digit4_kernel_read_readvariableop*
&savev2_digit4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_adacc_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop/
+savev2_adam_bn1_gamma_m_read_readvariableop.
*savev2_adam_bn1_beta_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop/
+savev2_adam_bn2_gamma_m_read_readvariableop.
*savev2_adam_bn2_beta_m_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop3
/savev2_adam_digit1_kernel_m_read_readvariableop1
-savev2_adam_digit1_bias_m_read_readvariableop3
/savev2_adam_digit2_kernel_m_read_readvariableop1
-savev2_adam_digit2_bias_m_read_readvariableop3
/savev2_adam_digit3_kernel_m_read_readvariableop1
-savev2_adam_digit3_bias_m_read_readvariableop3
/savev2_adam_digit4_kernel_m_read_readvariableop1
-savev2_adam_digit4_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop/
+savev2_adam_bn1_gamma_v_read_readvariableop.
*savev2_adam_bn1_beta_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableop/
+savev2_adam_bn2_gamma_v_read_readvariableop.
*savev2_adam_bn2_beta_v_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop3
/savev2_adam_digit1_kernel_v_read_readvariableop1
-savev2_adam_digit1_bias_v_read_readvariableop3
/savev2_adam_digit2_kernel_v_read_readvariableop1
-savev2_adam_digit2_bias_v_read_readvariableop3
/savev2_adam_digit3_kernel_v_read_readvariableop1
-savev2_adam_digit3_bias_v_read_readvariableop3
/savev2_adam_digit4_kernel_v_read_readvariableop1
-savev2_adam_digit4_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8885eca72109455a969ed9c990f46ffe/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?%
value?$B?$DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/adacc/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop$savev2_bn1_gamma_read_readvariableop#savev2_bn1_beta_read_readvariableop*savev2_bn1_moving_mean_read_readvariableop.savev2_bn1_moving_variance_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop$savev2_bn2_gamma_read_readvariableop#savev2_bn2_beta_read_readvariableop*savev2_bn2_moving_mean_read_readvariableop.savev2_bn2_moving_variance_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_digit1_kernel_read_readvariableop&savev2_digit1_bias_read_readvariableop(savev2_digit2_kernel_read_readvariableop&savev2_digit2_bias_read_readvariableop(savev2_digit3_kernel_read_readvariableop&savev2_digit3_bias_read_readvariableop(savev2_digit4_kernel_read_readvariableop&savev2_digit4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_adacc_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop+savev2_adam_bn1_gamma_m_read_readvariableop*savev2_adam_bn1_beta_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop+savev2_adam_bn2_gamma_m_read_readvariableop*savev2_adam_bn2_beta_m_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop/savev2_adam_digit1_kernel_m_read_readvariableop-savev2_adam_digit1_bias_m_read_readvariableop/savev2_adam_digit2_kernel_m_read_readvariableop-savev2_adam_digit2_bias_m_read_readvariableop/savev2_adam_digit3_kernel_m_read_readvariableop-savev2_adam_digit3_bias_m_read_readvariableop/savev2_adam_digit4_kernel_m_read_readvariableop-savev2_adam_digit4_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop+savev2_adam_bn1_gamma_v_read_readvariableop*savev2_adam_bn1_beta_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop+savev2_adam_bn2_gamma_v_read_readvariableop*savev2_adam_bn2_beta_v_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop/savev2_adam_digit1_kernel_v_read_readvariableop-savev2_adam_digit1_bias_v_read_readvariableop/savev2_adam_digit2_kernel_v_read_readvariableop-savev2_adam_digit2_bias_v_read_readvariableop/savev2_adam_digit3_kernel_v_read_readvariableop-savev2_adam_digit3_bias_v_read_readvariableop/savev2_adam_digit4_kernel_v_read_readvariableop-savev2_adam_digit4_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:@:@:@?:?:?:?:?:?:???:?:	?::	?::	?::	?:: : : : : : : : : : :@:@:@:@:@?:?:?:?:???:?:	?::	?::	?::	?::@:@:@:@:@?:?:?:?:???:?:	?::	?::	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:'#
!
_output_shapes
:???:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :,!(
&
_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:-%)
'
_output_shapes
:@?:!&

_output_shapes	
:?:!'

_output_shapes	
:?:!(

_output_shapes	
:?:')#
!
_output_shapes
:???:!*

_output_shapes	
:?:%+!

_output_shapes
:	?: ,

_output_shapes
::%-!

_output_shapes
:	?: .

_output_shapes
::%/!

_output_shapes
:	?: 0

_output_shapes
::%1!

_output_shapes
:	?: 2

_output_shapes
::,3(
&
_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@:-7)
'
_output_shapes
:@?:!8

_output_shapes	
:?:!9

_output_shapes	
:?:!:

_output_shapes	
:?:';#
!
_output_shapes
:???:!<

_output_shapes	
:?:%=!

_output_shapes
:	?: >

_output_shapes
::%?!

_output_shapes
:	?: @

_output_shapes
::%A!

_output_shapes
:	?: B

_output_shapes
::%C!

_output_shapes
:	?: D

_output_shapes
::E

_output_shapes
: 
?
?
3__inference_THSR_Captcha_Model_layer_call_fn_238453	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*+
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_2384062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????0?

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
?__inference_bn1_layer_call_and_return_conditional_losses_237845

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????.?@:@:@:@:@:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:?????????.?@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????.?@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:?????????.?@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_digit1_layer_call_and_return_conditional_losses_238148

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?$
?
?__inference_bn1_layer_call_and_return_conditional_losses_239178

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????.?@:@:@:@:@:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:?????????.?@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????.?@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:?????????.?@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
!__inference__wrapped_model_237464	
input;
7thsr_captcha_model_conv1_conv2d_readvariableop_resource<
8thsr_captcha_model_conv1_biasadd_readvariableop_resource2
.thsr_captcha_model_bn1_readvariableop_resource4
0thsr_captcha_model_bn1_readvariableop_1_resourceC
?thsr_captcha_model_bn1_fusedbatchnormv3_readvariableop_resourceE
Athsr_captcha_model_bn1_fusedbatchnormv3_readvariableop_1_resource;
7thsr_captcha_model_conv2_conv2d_readvariableop_resource<
8thsr_captcha_model_conv2_biasadd_readvariableop_resource2
.thsr_captcha_model_bn2_readvariableop_resource4
0thsr_captcha_model_bn2_readvariableop_1_resourceC
?thsr_captcha_model_bn2_fusedbatchnormv3_readvariableop_resourceE
Athsr_captcha_model_bn2_fusedbatchnormv3_readvariableop_1_resource<
8thsr_captcha_model_dense1_matmul_readvariableop_resource=
9thsr_captcha_model_dense1_biasadd_readvariableop_resource<
8thsr_captcha_model_digit4_matmul_readvariableop_resource=
9thsr_captcha_model_digit4_biasadd_readvariableop_resource<
8thsr_captcha_model_digit3_matmul_readvariableop_resource=
9thsr_captcha_model_digit3_biasadd_readvariableop_resource<
8thsr_captcha_model_digit2_matmul_readvariableop_resource=
9thsr_captcha_model_digit2_biasadd_readvariableop_resource<
8thsr_captcha_model_digit1_matmul_readvariableop_resource=
9thsr_captcha_model_digit1_biasadd_readvariableop_resource
identity?y
 THSR_Captcha_Model/reshape/ShapeShapeinput*
T0*
_output_shapes
:2"
 THSR_Captcha_Model/reshape/Shape?
.THSR_Captcha_Model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.THSR_Captcha_Model/reshape/strided_slice/stack?
0THSR_Captcha_Model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0THSR_Captcha_Model/reshape/strided_slice/stack_1?
0THSR_Captcha_Model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0THSR_Captcha_Model/reshape/strided_slice/stack_2?
(THSR_Captcha_Model/reshape/strided_sliceStridedSlice)THSR_Captcha_Model/reshape/Shape:output:07THSR_Captcha_Model/reshape/strided_slice/stack:output:09THSR_Captcha_Model/reshape/strided_slice/stack_1:output:09THSR_Captcha_Model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(THSR_Captcha_Model/reshape/strided_slice?
*THSR_Captcha_Model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :02,
*THSR_Captcha_Model/reshape/Reshape/shape/1?
*THSR_Captcha_Model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2,
*THSR_Captcha_Model/reshape/Reshape/shape/2?
*THSR_Captcha_Model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2,
*THSR_Captcha_Model/reshape/Reshape/shape/3?
(THSR_Captcha_Model/reshape/Reshape/shapePack1THSR_Captcha_Model/reshape/strided_slice:output:03THSR_Captcha_Model/reshape/Reshape/shape/1:output:03THSR_Captcha_Model/reshape/Reshape/shape/2:output:03THSR_Captcha_Model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2*
(THSR_Captcha_Model/reshape/Reshape/shape?
"THSR_Captcha_Model/reshape/ReshapeReshapeinput1THSR_Captcha_Model/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????0?2$
"THSR_Captcha_Model/reshape/Reshape?
.THSR_Captcha_Model/conv1/Conv2D/ReadVariableOpReadVariableOp7thsr_captcha_model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype020
.THSR_Captcha_Model/conv1/Conv2D/ReadVariableOp?
THSR_Captcha_Model/conv1/Conv2DConv2D+THSR_Captcha_Model/reshape/Reshape:output:06THSR_Captcha_Model/conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????.?@*
paddingVALID*
strides
2!
THSR_Captcha_Model/conv1/Conv2D?
/THSR_Captcha_Model/conv1/BiasAdd/ReadVariableOpReadVariableOp8thsr_captcha_model_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/THSR_Captcha_Model/conv1/BiasAdd/ReadVariableOp?
 THSR_Captcha_Model/conv1/BiasAddBiasAdd(THSR_Captcha_Model/conv1/Conv2D:output:07THSR_Captcha_Model/conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????.?@2"
 THSR_Captcha_Model/conv1/BiasAdd?
THSR_Captcha_Model/conv1/ReluRelu)THSR_Captcha_Model/conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????.?@2
THSR_Captcha_Model/conv1/Relu?
%THSR_Captcha_Model/bn1/ReadVariableOpReadVariableOp.thsr_captcha_model_bn1_readvariableop_resource*
_output_shapes
:@*
dtype02'
%THSR_Captcha_Model/bn1/ReadVariableOp?
'THSR_Captcha_Model/bn1/ReadVariableOp_1ReadVariableOp0thsr_captcha_model_bn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'THSR_Captcha_Model/bn1/ReadVariableOp_1?
6THSR_Captcha_Model/bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp?thsr_captcha_model_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6THSR_Captcha_Model/bn1/FusedBatchNormV3/ReadVariableOp?
8THSR_Captcha_Model/bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAthsr_captcha_model_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8THSR_Captcha_Model/bn1/FusedBatchNormV3/ReadVariableOp_1?
'THSR_Captcha_Model/bn1/FusedBatchNormV3FusedBatchNormV3+THSR_Captcha_Model/conv1/Relu:activations:0-THSR_Captcha_Model/bn1/ReadVariableOp:value:0/THSR_Captcha_Model/bn1/ReadVariableOp_1:value:0>THSR_Captcha_Model/bn1/FusedBatchNormV3/ReadVariableOp:value:0@THSR_Captcha_Model/bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????.?@:@:@:@:@:*
epsilon%??'7*
is_training( 2)
'THSR_Captcha_Model/bn1/FusedBatchNormV3?
#THSR_Captcha_Model/pooling1/MaxPoolMaxPool+THSR_Captcha_Model/bn1/FusedBatchNormV3:y:0*/
_output_shapes
:?????????E@*
ksize
*
paddingVALID*
strides
2%
#THSR_Captcha_Model/pooling1/MaxPool?
.THSR_Captcha_Model/conv2/Conv2D/ReadVariableOpReadVariableOp7thsr_captcha_model_conv2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype020
.THSR_Captcha_Model/conv2/Conv2D/ReadVariableOp?
THSR_Captcha_Model/conv2/Conv2DConv2D,THSR_Captcha_Model/pooling1/MaxPool:output:06THSR_Captcha_Model/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????C?*
paddingVALID*
strides
2!
THSR_Captcha_Model/conv2/Conv2D?
/THSR_Captcha_Model/conv2/BiasAdd/ReadVariableOpReadVariableOp8thsr_captcha_model_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/THSR_Captcha_Model/conv2/BiasAdd/ReadVariableOp?
 THSR_Captcha_Model/conv2/BiasAddBiasAdd(THSR_Captcha_Model/conv2/Conv2D:output:07THSR_Captcha_Model/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????C?2"
 THSR_Captcha_Model/conv2/BiasAdd?
THSR_Captcha_Model/conv2/ReluRelu)THSR_Captcha_Model/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????C?2
THSR_Captcha_Model/conv2/Relu?
%THSR_Captcha_Model/bn2/ReadVariableOpReadVariableOp.thsr_captcha_model_bn2_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%THSR_Captcha_Model/bn2/ReadVariableOp?
'THSR_Captcha_Model/bn2/ReadVariableOp_1ReadVariableOp0thsr_captcha_model_bn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'THSR_Captcha_Model/bn2/ReadVariableOp_1?
6THSR_Captcha_Model/bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp?thsr_captcha_model_bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6THSR_Captcha_Model/bn2/FusedBatchNormV3/ReadVariableOp?
8THSR_Captcha_Model/bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAthsr_captcha_model_bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8THSR_Captcha_Model/bn2/FusedBatchNormV3/ReadVariableOp_1?
'THSR_Captcha_Model/bn2/FusedBatchNormV3FusedBatchNormV3+THSR_Captcha_Model/conv2/Relu:activations:0-THSR_Captcha_Model/bn2/ReadVariableOp:value:0/THSR_Captcha_Model/bn2/ReadVariableOp_1:value:0>THSR_Captcha_Model/bn2/FusedBatchNormV3/ReadVariableOp:value:0@THSR_Captcha_Model/bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????C?:?:?:?:?:*
epsilon%??'7*
is_training( 2)
'THSR_Captcha_Model/bn2/FusedBatchNormV3?
#THSR_Captcha_Model/pooling2/MaxPoolMaxPool+THSR_Captcha_Model/bn2/FusedBatchNormV3:y:0*0
_output_shapes
:?????????
!?*
ksize
*
paddingVALID*
strides
2%
#THSR_Captcha_Model/pooling2/MaxPool?
 THSR_Captcha_Model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2"
 THSR_Captcha_Model/flatten/Const?
"THSR_Captcha_Model/flatten/ReshapeReshape,THSR_Captcha_Model/pooling2/MaxPool:output:0)THSR_Captcha_Model/flatten/Const:output:0*
T0*)
_output_shapes
:???????????2$
"THSR_Captcha_Model/flatten/Reshape?
/THSR_Captcha_Model/dense1/MatMul/ReadVariableOpReadVariableOp8thsr_captcha_model_dense1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype021
/THSR_Captcha_Model/dense1/MatMul/ReadVariableOp?
 THSR_Captcha_Model/dense1/MatMulMatMul+THSR_Captcha_Model/flatten/Reshape:output:07THSR_Captcha_Model/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 THSR_Captcha_Model/dense1/MatMul?
0THSR_Captcha_Model/dense1/BiasAdd/ReadVariableOpReadVariableOp9thsr_captcha_model_dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0THSR_Captcha_Model/dense1/BiasAdd/ReadVariableOp?
!THSR_Captcha_Model/dense1/BiasAddBiasAdd*THSR_Captcha_Model/dense1/MatMul:product:08THSR_Captcha_Model/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!THSR_Captcha_Model/dense1/BiasAdd?
&THSR_Captcha_Model/dropout0.5/IdentityIdentity*THSR_Captcha_Model/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&THSR_Captcha_Model/dropout0.5/Identity?
/THSR_Captcha_Model/digit4/MatMul/ReadVariableOpReadVariableOp8thsr_captcha_model_digit4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/THSR_Captcha_Model/digit4/MatMul/ReadVariableOp?
 THSR_Captcha_Model/digit4/MatMulMatMul/THSR_Captcha_Model/dropout0.5/Identity:output:07THSR_Captcha_Model/digit4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 THSR_Captcha_Model/digit4/MatMul?
0THSR_Captcha_Model/digit4/BiasAdd/ReadVariableOpReadVariableOp9thsr_captcha_model_digit4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0THSR_Captcha_Model/digit4/BiasAdd/ReadVariableOp?
!THSR_Captcha_Model/digit4/BiasAddBiasAdd*THSR_Captcha_Model/digit4/MatMul:product:08THSR_Captcha_Model/digit4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!THSR_Captcha_Model/digit4/BiasAdd?
/THSR_Captcha_Model/digit3/MatMul/ReadVariableOpReadVariableOp8thsr_captcha_model_digit3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/THSR_Captcha_Model/digit3/MatMul/ReadVariableOp?
 THSR_Captcha_Model/digit3/MatMulMatMul/THSR_Captcha_Model/dropout0.5/Identity:output:07THSR_Captcha_Model/digit3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 THSR_Captcha_Model/digit3/MatMul?
0THSR_Captcha_Model/digit3/BiasAdd/ReadVariableOpReadVariableOp9thsr_captcha_model_digit3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0THSR_Captcha_Model/digit3/BiasAdd/ReadVariableOp?
!THSR_Captcha_Model/digit3/BiasAddBiasAdd*THSR_Captcha_Model/digit3/MatMul:product:08THSR_Captcha_Model/digit3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!THSR_Captcha_Model/digit3/BiasAdd?
/THSR_Captcha_Model/digit2/MatMul/ReadVariableOpReadVariableOp8thsr_captcha_model_digit2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/THSR_Captcha_Model/digit2/MatMul/ReadVariableOp?
 THSR_Captcha_Model/digit2/MatMulMatMul/THSR_Captcha_Model/dropout0.5/Identity:output:07THSR_Captcha_Model/digit2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 THSR_Captcha_Model/digit2/MatMul?
0THSR_Captcha_Model/digit2/BiasAdd/ReadVariableOpReadVariableOp9thsr_captcha_model_digit2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0THSR_Captcha_Model/digit2/BiasAdd/ReadVariableOp?
!THSR_Captcha_Model/digit2/BiasAddBiasAdd*THSR_Captcha_Model/digit2/MatMul:product:08THSR_Captcha_Model/digit2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!THSR_Captcha_Model/digit2/BiasAdd?
/THSR_Captcha_Model/digit1/MatMul/ReadVariableOpReadVariableOp8thsr_captcha_model_digit1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/THSR_Captcha_Model/digit1/MatMul/ReadVariableOp?
 THSR_Captcha_Model/digit1/MatMulMatMul/THSR_Captcha_Model/dropout0.5/Identity:output:07THSR_Captcha_Model/digit1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 THSR_Captcha_Model/digit1/MatMul?
0THSR_Captcha_Model/digit1/BiasAdd/ReadVariableOpReadVariableOp9thsr_captcha_model_digit1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0THSR_Captcha_Model/digit1/BiasAdd/ReadVariableOp?
!THSR_Captcha_Model/digit1/BiasAddBiasAdd*THSR_Captcha_Model/digit1/MatMul:product:08THSR_Captcha_Model/digit1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!THSR_Captcha_Model/digit1/BiasAdd?
'THSR_Captcha_Model/reshape_digit1/ShapeShape*THSR_Captcha_Model/digit1/BiasAdd:output:0*
T0*
_output_shapes
:2)
'THSR_Captcha_Model/reshape_digit1/Shape?
5THSR_Captcha_Model/reshape_digit1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5THSR_Captcha_Model/reshape_digit1/strided_slice/stack?
7THSR_Captcha_Model/reshape_digit1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit1/strided_slice/stack_1?
7THSR_Captcha_Model/reshape_digit1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit1/strided_slice/stack_2?
/THSR_Captcha_Model/reshape_digit1/strided_sliceStridedSlice0THSR_Captcha_Model/reshape_digit1/Shape:output:0>THSR_Captcha_Model/reshape_digit1/strided_slice/stack:output:0@THSR_Captcha_Model/reshape_digit1/strided_slice/stack_1:output:0@THSR_Captcha_Model/reshape_digit1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/THSR_Captcha_Model/reshape_digit1/strided_slice?
1THSR_Captcha_Model/reshape_digit1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit1/Reshape/shape/1?
1THSR_Captcha_Model/reshape_digit1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit1/Reshape/shape/2?
/THSR_Captcha_Model/reshape_digit1/Reshape/shapePack8THSR_Captcha_Model/reshape_digit1/strided_slice:output:0:THSR_Captcha_Model/reshape_digit1/Reshape/shape/1:output:0:THSR_Captcha_Model/reshape_digit1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:21
/THSR_Captcha_Model/reshape_digit1/Reshape/shape?
)THSR_Captcha_Model/reshape_digit1/ReshapeReshape*THSR_Captcha_Model/digit1/BiasAdd:output:08THSR_Captcha_Model/reshape_digit1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2+
)THSR_Captcha_Model/reshape_digit1/Reshape?
'THSR_Captcha_Model/reshape_digit2/ShapeShape*THSR_Captcha_Model/digit2/BiasAdd:output:0*
T0*
_output_shapes
:2)
'THSR_Captcha_Model/reshape_digit2/Shape?
5THSR_Captcha_Model/reshape_digit2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5THSR_Captcha_Model/reshape_digit2/strided_slice/stack?
7THSR_Captcha_Model/reshape_digit2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit2/strided_slice/stack_1?
7THSR_Captcha_Model/reshape_digit2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit2/strided_slice/stack_2?
/THSR_Captcha_Model/reshape_digit2/strided_sliceStridedSlice0THSR_Captcha_Model/reshape_digit2/Shape:output:0>THSR_Captcha_Model/reshape_digit2/strided_slice/stack:output:0@THSR_Captcha_Model/reshape_digit2/strided_slice/stack_1:output:0@THSR_Captcha_Model/reshape_digit2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/THSR_Captcha_Model/reshape_digit2/strided_slice?
1THSR_Captcha_Model/reshape_digit2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit2/Reshape/shape/1?
1THSR_Captcha_Model/reshape_digit2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit2/Reshape/shape/2?
/THSR_Captcha_Model/reshape_digit2/Reshape/shapePack8THSR_Captcha_Model/reshape_digit2/strided_slice:output:0:THSR_Captcha_Model/reshape_digit2/Reshape/shape/1:output:0:THSR_Captcha_Model/reshape_digit2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:21
/THSR_Captcha_Model/reshape_digit2/Reshape/shape?
)THSR_Captcha_Model/reshape_digit2/ReshapeReshape*THSR_Captcha_Model/digit2/BiasAdd:output:08THSR_Captcha_Model/reshape_digit2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2+
)THSR_Captcha_Model/reshape_digit2/Reshape?
'THSR_Captcha_Model/reshape_digit3/ShapeShape*THSR_Captcha_Model/digit3/BiasAdd:output:0*
T0*
_output_shapes
:2)
'THSR_Captcha_Model/reshape_digit3/Shape?
5THSR_Captcha_Model/reshape_digit3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5THSR_Captcha_Model/reshape_digit3/strided_slice/stack?
7THSR_Captcha_Model/reshape_digit3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit3/strided_slice/stack_1?
7THSR_Captcha_Model/reshape_digit3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit3/strided_slice/stack_2?
/THSR_Captcha_Model/reshape_digit3/strided_sliceStridedSlice0THSR_Captcha_Model/reshape_digit3/Shape:output:0>THSR_Captcha_Model/reshape_digit3/strided_slice/stack:output:0@THSR_Captcha_Model/reshape_digit3/strided_slice/stack_1:output:0@THSR_Captcha_Model/reshape_digit3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/THSR_Captcha_Model/reshape_digit3/strided_slice?
1THSR_Captcha_Model/reshape_digit3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit3/Reshape/shape/1?
1THSR_Captcha_Model/reshape_digit3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit3/Reshape/shape/2?
/THSR_Captcha_Model/reshape_digit3/Reshape/shapePack8THSR_Captcha_Model/reshape_digit3/strided_slice:output:0:THSR_Captcha_Model/reshape_digit3/Reshape/shape/1:output:0:THSR_Captcha_Model/reshape_digit3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:21
/THSR_Captcha_Model/reshape_digit3/Reshape/shape?
)THSR_Captcha_Model/reshape_digit3/ReshapeReshape*THSR_Captcha_Model/digit3/BiasAdd:output:08THSR_Captcha_Model/reshape_digit3/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2+
)THSR_Captcha_Model/reshape_digit3/Reshape?
'THSR_Captcha_Model/reshape_digit4/ShapeShape*THSR_Captcha_Model/digit4/BiasAdd:output:0*
T0*
_output_shapes
:2)
'THSR_Captcha_Model/reshape_digit4/Shape?
5THSR_Captcha_Model/reshape_digit4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5THSR_Captcha_Model/reshape_digit4/strided_slice/stack?
7THSR_Captcha_Model/reshape_digit4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit4/strided_slice/stack_1?
7THSR_Captcha_Model/reshape_digit4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7THSR_Captcha_Model/reshape_digit4/strided_slice/stack_2?
/THSR_Captcha_Model/reshape_digit4/strided_sliceStridedSlice0THSR_Captcha_Model/reshape_digit4/Shape:output:0>THSR_Captcha_Model/reshape_digit4/strided_slice/stack:output:0@THSR_Captcha_Model/reshape_digit4/strided_slice/stack_1:output:0@THSR_Captcha_Model/reshape_digit4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/THSR_Captcha_Model/reshape_digit4/strided_slice?
1THSR_Captcha_Model/reshape_digit4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit4/Reshape/shape/1?
1THSR_Captcha_Model/reshape_digit4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1THSR_Captcha_Model/reshape_digit4/Reshape/shape/2?
/THSR_Captcha_Model/reshape_digit4/Reshape/shapePack8THSR_Captcha_Model/reshape_digit4/strided_slice:output:0:THSR_Captcha_Model/reshape_digit4/Reshape/shape/1:output:0:THSR_Captcha_Model/reshape_digit4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:21
/THSR_Captcha_Model/reshape_digit4/Reshape/shape?
)THSR_Captcha_Model/reshape_digit4/ReshapeReshape*THSR_Captcha_Model/digit4/BiasAdd:output:08THSR_Captcha_Model/reshape_digit4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2+
)THSR_Captcha_Model/reshape_digit4/Reshape?
)THSR_Captcha_Model/prediction/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)THSR_Captcha_Model/prediction/concat/axis?
$THSR_Captcha_Model/prediction/concatConcatV22THSR_Captcha_Model/reshape_digit1/Reshape:output:02THSR_Captcha_Model/reshape_digit2/Reshape:output:02THSR_Captcha_Model/reshape_digit3/Reshape:output:02THSR_Captcha_Model/reshape_digit4/Reshape:output:02THSR_Captcha_Model/prediction/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????2&
$THSR_Captcha_Model/prediction/concat?
IdentityIdentity-THSR_Captcha_Model/prediction/concat:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?:::::::::::::::::::::::S O
,
_output_shapes
:?????????0?

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ϝ
?!
"__inference__traced_restore_240053
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias 
assignvariableop_2_bn1_gamma
assignvariableop_3_bn1_beta&
"assignvariableop_4_bn1_moving_mean*
&assignvariableop_5_bn1_moving_variance#
assignvariableop_6_conv2_kernel!
assignvariableop_7_conv2_bias 
assignvariableop_8_bn2_gamma
assignvariableop_9_bn2_beta'
#assignvariableop_10_bn2_moving_mean+
'assignvariableop_11_bn2_moving_variance%
!assignvariableop_12_dense1_kernel#
assignvariableop_13_dense1_bias%
!assignvariableop_14_digit1_kernel#
assignvariableop_15_digit1_bias%
!assignvariableop_16_digit2_kernel#
assignvariableop_17_digit2_bias%
!assignvariableop_18_digit3_kernel#
assignvariableop_19_digit3_bias%
!assignvariableop_20_digit4_kernel#
assignvariableop_21_digit4_bias!
assignvariableop_22_adam_iter#
assignvariableop_23_adam_beta_1#
assignvariableop_24_adam_beta_2"
assignvariableop_25_adam_decay*
&assignvariableop_26_adam_learning_rate
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1
assignvariableop_31_adacc+
'assignvariableop_32_adam_conv1_kernel_m)
%assignvariableop_33_adam_conv1_bias_m(
$assignvariableop_34_adam_bn1_gamma_m'
#assignvariableop_35_adam_bn1_beta_m+
'assignvariableop_36_adam_conv2_kernel_m)
%assignvariableop_37_adam_conv2_bias_m(
$assignvariableop_38_adam_bn2_gamma_m'
#assignvariableop_39_adam_bn2_beta_m,
(assignvariableop_40_adam_dense1_kernel_m*
&assignvariableop_41_adam_dense1_bias_m,
(assignvariableop_42_adam_digit1_kernel_m*
&assignvariableop_43_adam_digit1_bias_m,
(assignvariableop_44_adam_digit2_kernel_m*
&assignvariableop_45_adam_digit2_bias_m,
(assignvariableop_46_adam_digit3_kernel_m*
&assignvariableop_47_adam_digit3_bias_m,
(assignvariableop_48_adam_digit4_kernel_m*
&assignvariableop_49_adam_digit4_bias_m+
'assignvariableop_50_adam_conv1_kernel_v)
%assignvariableop_51_adam_conv1_bias_v(
$assignvariableop_52_adam_bn1_gamma_v'
#assignvariableop_53_adam_bn1_beta_v+
'assignvariableop_54_adam_conv2_kernel_v)
%assignvariableop_55_adam_conv2_bias_v(
$assignvariableop_56_adam_bn2_gamma_v'
#assignvariableop_57_adam_bn2_beta_v,
(assignvariableop_58_adam_dense1_kernel_v*
&assignvariableop_59_adam_dense1_bias_v,
(assignvariableop_60_adam_digit1_kernel_v*
&assignvariableop_61_adam_digit1_bias_v,
(assignvariableop_62_adam_digit2_kernel_v*
&assignvariableop_63_adam_digit2_bias_v,
(assignvariableop_64_adam_digit3_kernel_v*
&assignvariableop_65_adam_digit3_bias_v,
(assignvariableop_66_adam_digit4_kernel_v*
&assignvariableop_67_adam_digit4_bias_v
identity_69??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?%
value?$B?$DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/adacc/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn1_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn1_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_bn1_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_bn1_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn2_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn2_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_bn2_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_bn2_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense1_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense1_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_digit1_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_digit1_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_digit2_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_digit2_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_digit3_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_digit3_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_digit4_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_digit4_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0	*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_adaccIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_conv1_kernel_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_conv1_bias_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp$assignvariableop_34_adam_bn1_gamma_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp#assignvariableop_35_adam_bn1_beta_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_conv2_kernel_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_conv2_bias_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_adam_bn2_gamma_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp#assignvariableop_39_adam_bn2_beta_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense1_kernel_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp&assignvariableop_41_adam_dense1_bias_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_digit1_kernel_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp&assignvariableop_43_adam_digit1_bias_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_digit2_kernel_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_digit2_bias_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_digit3_kernel_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp&assignvariableop_47_adam_digit3_bias_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_digit4_kernel_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp&assignvariableop_49_adam_digit4_bias_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_conv1_kernel_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp%assignvariableop_51_adam_conv1_bias_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp$assignvariableop_52_adam_bn1_gamma_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp#assignvariableop_53_adam_bn1_beta_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_conv2_kernel_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp%assignvariableop_55_adam_conv2_bias_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp$assignvariableop_56_adam_bn2_gamma_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp#assignvariableop_57_adam_bn2_beta_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense1_kernel_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_dense1_bias_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_digit1_kernel_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp&assignvariableop_61_adam_digit1_bias_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_digit2_kernel_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp&assignvariableop_63_adam_digit2_bias_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_digit3_kernel_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp&assignvariableop_65_adam_digit3_bias_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_digit4_kernel_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp&assignvariableop_67_adam_digit4_bias_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_68Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_68?
Identity_69IdentityIdentity_68:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_69"#
identity_69Identity_69:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: 
?
d
+__inference_dropout0.5_layer_call_fn_239436

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout0.5_layer_call_and_return_conditional_losses_2380422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_239548

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
(__inference_reshape_layer_call_fn_239060

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:?????????0?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2378022
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????0?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????0?:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs
?$
?
?__inference_bn2_layer_call_and_return_conditional_losses_239340

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_dropout0.5_layer_call_fn_239441

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout0.5_layer_call_and_return_conditional_losses_2380472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_digit1_layer_call_and_return_conditional_losses_239451

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_238240

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_prediction_layer_call_and_return_conditional_losses_238257

inputs
inputs_1
inputs_2
inputs_3
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:?????????:?????????:?????????:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_conv2_layer_call_fn_237646

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2376362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?M
?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238522

inputs
conv1_238459
conv1_238461

bn1_238464

bn1_238466

bn1_238468

bn1_238470
conv2_238474
conv2_238476

bn2_238479

bn2_238481

bn2_238483

bn2_238485
dense1_238490
dense1_238492
digit4_238496
digit4_238498
digit3_238501
digit3_238503
digit2_238506
digit2_238508
digit1_238511
digit1_238513
identity??bn1/StatefulPartitionedCall?bn2/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?digit1/StatefulPartitionedCall?digit2/StatefulPartitionedCall?digit3/StatefulPartitionedCall?digit4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:?????????0?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2378022
reshape/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_238459conv1_238461*
Tin
2*
Tout
2*0
_output_shapes
:?????????.?@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2374762
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
bn1_238464
bn1_238466
bn1_238468
bn1_238470*
Tin	
2*
Tout
2*0
_output_shapes
:?????????.?@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2378632
bn1/StatefulPartitionedCall?
pooling1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????E@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling1_layer_call_and_return_conditional_losses_2376182
pooling1/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall!pooling1/PartitionedCall:output:0conv2_238474conv2_238476*
Tin
2*
Tout
2*0
_output_shapes
:?????????C?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2376362
conv2/StatefulPartitionedCall?
bn2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0
bn2_238479
bn2_238481
bn2_238483
bn2_238485*
Tin	
2*
Tout
2*0
_output_shapes
:?????????C?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2379532
bn2/StatefulPartitionedCall?
pooling2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????
!?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling2_layer_call_and_return_conditional_losses_2377782
pooling2/PartitionedCall?
flatten/PartitionedCallPartitionedCall!pooling2/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2379962
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_238490dense1_238492*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_2380142 
dense1/StatefulPartitionedCall?
dropout0.5/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout0.5_layer_call_and_return_conditional_losses_2380472
dropout0.5/PartitionedCall?
digit4/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit4_238496digit4_238498*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit4_layer_call_and_return_conditional_losses_2380702 
digit4/StatefulPartitionedCall?
digit3/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit3_238501digit3_238503*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit3_layer_call_and_return_conditional_losses_2380962 
digit3/StatefulPartitionedCall?
digit2/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit2_238506digit2_238508*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit2_layer_call_and_return_conditional_losses_2381222 
digit2/StatefulPartitionedCall?
digit1/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit1_238511digit1_238513*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit1_layer_call_and_return_conditional_losses_2381482 
digit1/StatefulPartitionedCall?
reshape_digit1/PartitionedCallPartitionedCall'digit1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_2381772 
reshape_digit1/PartitionedCall?
reshape_digit2/PartitionedCallPartitionedCall'digit2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_2381982 
reshape_digit2/PartitionedCall?
reshape_digit3/PartitionedCallPartitionedCall'digit3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_2382192 
reshape_digit3/PartitionedCall?
reshape_digit4/PartitionedCallPartitionedCall'digit4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_2382402 
reshape_digit4/PartitionedCall?
prediction/PartitionedCallPartitionedCall'reshape_digit1/PartitionedCall:output:0'reshape_digit2/PartitionedCall:output:0'reshape_digit3/PartitionedCall:output:0'reshape_digit4/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_2382572
prediction/PartitionedCall?
IdentityIdentity#prediction/PartitionedCall:output:0^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^digit1/StatefulPartitionedCall^digit2/StatefulPartitionedCall^digit3/StatefulPartitionedCall^digit4/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
digit1/StatefulPartitionedCalldigit1/StatefulPartitionedCall2@
digit2/StatefulPartitionedCalldigit2/StatefulPartitionedCall2@
digit3/StatefulPartitionedCalldigit3/StatefulPartitionedCall2@
digit4/StatefulPartitionedCalldigit4/StatefulPartitionedCall:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?O
?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238269	
input
conv1_237810
conv1_237812

bn1_237890

bn1_237892

bn1_237894

bn1_237896
conv2_237900
conv2_237902

bn2_237980

bn2_237982

bn2_237984

bn2_237986
dense1_238025
dense1_238027
digit4_238081
digit4_238083
digit3_238107
digit3_238109
digit2_238133
digit2_238135
digit1_238159
digit1_238161
identity??bn1/StatefulPartitionedCall?bn2/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?digit1/StatefulPartitionedCall?digit2/StatefulPartitionedCall?digit3/StatefulPartitionedCall?digit4/StatefulPartitionedCall?"dropout0.5/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*0
_output_shapes
:?????????0?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2378022
reshape/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_237810conv1_237812*
Tin
2*
Tout
2*0
_output_shapes
:?????????.?@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2374762
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
bn1_237890
bn1_237892
bn1_237894
bn1_237896*
Tin	
2*
Tout
2*0
_output_shapes
:?????????.?@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2378452
bn1/StatefulPartitionedCall?
pooling1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????E@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling1_layer_call_and_return_conditional_losses_2376182
pooling1/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall!pooling1/PartitionedCall:output:0conv2_237900conv2_237902*
Tin
2*
Tout
2*0
_output_shapes
:?????????C?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2376362
conv2/StatefulPartitionedCall?
bn2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0
bn2_237980
bn2_237982
bn2_237984
bn2_237986*
Tin	
2*
Tout
2*0
_output_shapes
:?????????C?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2379352
bn2/StatefulPartitionedCall?
pooling2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????
!?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling2_layer_call_and_return_conditional_losses_2377782
pooling2/PartitionedCall?
flatten/PartitionedCallPartitionedCall!pooling2/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2379962
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_238025dense1_238027*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_2380142 
dense1/StatefulPartitionedCall?
"dropout0.5/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout0.5_layer_call_and_return_conditional_losses_2380422$
"dropout0.5/StatefulPartitionedCall?
digit4/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit4_238081digit4_238083*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit4_layer_call_and_return_conditional_losses_2380702 
digit4/StatefulPartitionedCall?
digit3/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit3_238107digit3_238109*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit3_layer_call_and_return_conditional_losses_2380962 
digit3/StatefulPartitionedCall?
digit2/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit2_238133digit2_238135*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit2_layer_call_and_return_conditional_losses_2381222 
digit2/StatefulPartitionedCall?
digit1/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit1_238159digit1_238161*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit1_layer_call_and_return_conditional_losses_2381482 
digit1/StatefulPartitionedCall?
reshape_digit1/PartitionedCallPartitionedCall'digit1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_2381772 
reshape_digit1/PartitionedCall?
reshape_digit2/PartitionedCallPartitionedCall'digit2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_2381982 
reshape_digit2/PartitionedCall?
reshape_digit3/PartitionedCallPartitionedCall'digit3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_2382192 
reshape_digit3/PartitionedCall?
reshape_digit4/PartitionedCallPartitionedCall'digit4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_2382402 
reshape_digit4/PartitionedCall?
prediction/PartitionedCallPartitionedCall'reshape_digit1/PartitionedCall:output:0'reshape_digit2/PartitionedCall:output:0'reshape_digit3/PartitionedCall:output:0'reshape_digit4/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_2382572
prediction/PartitionedCall?
IdentityIdentity#prediction/PartitionedCall:output:0^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^digit1/StatefulPartitionedCall^digit2/StatefulPartitionedCall^digit3/StatefulPartitionedCall^digit4/StatefulPartitionedCall#^dropout0.5/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
digit1/StatefulPartitionedCalldigit1/StatefulPartitionedCall2@
digit2/StatefulPartitionedCalldigit2/StatefulPartitionedCall2@
digit3/StatefulPartitionedCalldigit3/StatefulPartitionedCall2@
digit4/StatefulPartitionedCalldigit4/StatefulPartitionedCall2H
"dropout0.5/StatefulPartitionedCall"dropout0.5/StatefulPartitionedCall:S O
,
_output_shapes
:?????????0?

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_bn1_layer_call_and_return_conditional_losses_239196

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????.?@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????.?@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????.?@:::::X T
0
_output_shapes
:?????????.?@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
?__inference_bn2_layer_call_and_return_conditional_losses_237935

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????C?:?:?:?:?:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:?????????C?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????C?::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:?????????C?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_239530

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?O
?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238406

inputs
conv1_238343
conv1_238345

bn1_238348

bn1_238350

bn1_238352

bn1_238354
conv2_238358
conv2_238360

bn2_238363

bn2_238365

bn2_238367

bn2_238369
dense1_238374
dense1_238376
digit4_238380
digit4_238382
digit3_238385
digit3_238387
digit2_238390
digit2_238392
digit1_238395
digit1_238397
identity??bn1/StatefulPartitionedCall?bn2/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?digit1/StatefulPartitionedCall?digit2/StatefulPartitionedCall?digit3/StatefulPartitionedCall?digit4/StatefulPartitionedCall?"dropout0.5/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:?????????0?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2378022
reshape/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_238343conv1_238345*
Tin
2*
Tout
2*0
_output_shapes
:?????????.?@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2374762
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
bn1_238348
bn1_238350
bn1_238352
bn1_238354*
Tin	
2*
Tout
2*0
_output_shapes
:?????????.?@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2378452
bn1/StatefulPartitionedCall?
pooling1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????E@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling1_layer_call_and_return_conditional_losses_2376182
pooling1/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall!pooling1/PartitionedCall:output:0conv2_238358conv2_238360*
Tin
2*
Tout
2*0
_output_shapes
:?????????C?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2376362
conv2/StatefulPartitionedCall?
bn2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0
bn2_238363
bn2_238365
bn2_238367
bn2_238369*
Tin	
2*
Tout
2*0
_output_shapes
:?????????C?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2379352
bn2/StatefulPartitionedCall?
pooling2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????
!?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling2_layer_call_and_return_conditional_losses_2377782
pooling2/PartitionedCall?
flatten/PartitionedCallPartitionedCall!pooling2/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2379962
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_238374dense1_238376*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_2380142 
dense1/StatefulPartitionedCall?
"dropout0.5/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout0.5_layer_call_and_return_conditional_losses_2380422$
"dropout0.5/StatefulPartitionedCall?
digit4/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit4_238380digit4_238382*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit4_layer_call_and_return_conditional_losses_2380702 
digit4/StatefulPartitionedCall?
digit3/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit3_238385digit3_238387*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit3_layer_call_and_return_conditional_losses_2380962 
digit3/StatefulPartitionedCall?
digit2/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit2_238390digit2_238392*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit2_layer_call_and_return_conditional_losses_2381222 
digit2/StatefulPartitionedCall?
digit1/StatefulPartitionedCallStatefulPartitionedCall+dropout0.5/StatefulPartitionedCall:output:0digit1_238395digit1_238397*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit1_layer_call_and_return_conditional_losses_2381482 
digit1/StatefulPartitionedCall?
reshape_digit1/PartitionedCallPartitionedCall'digit1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_2381772 
reshape_digit1/PartitionedCall?
reshape_digit2/PartitionedCallPartitionedCall'digit2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_2381982 
reshape_digit2/PartitionedCall?
reshape_digit3/PartitionedCallPartitionedCall'digit3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_2382192 
reshape_digit3/PartitionedCall?
reshape_digit4/PartitionedCallPartitionedCall'digit4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_2382402 
reshape_digit4/PartitionedCall?
prediction/PartitionedCallPartitionedCall'reshape_digit1/PartitionedCall:output:0'reshape_digit2/PartitionedCall:output:0'reshape_digit3/PartitionedCall:output:0'reshape_digit4/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_2382572
prediction/PartitionedCall?
IdentityIdentity#prediction/PartitionedCall:output:0^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^digit1/StatefulPartitionedCall^digit2/StatefulPartitionedCall^digit3/StatefulPartitionedCall^digit4/StatefulPartitionedCall#^dropout0.5/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
digit1/StatefulPartitionedCalldigit1/StatefulPartitionedCall2@
digit2/StatefulPartitionedCalldigit2/StatefulPartitionedCall2@
digit3/StatefulPartitionedCalldigit3/StatefulPartitionedCall2@
digit4/StatefulPartitionedCalldigit4/StatefulPartitionedCall2H
"dropout0.5/StatefulPartitionedCall"dropout0.5/StatefulPartitionedCall:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
D
(__inference_flatten_layer_call_fn_239395

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2379962
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????
!?:X T
0
_output_shapes
:?????????
!?
 
_user_specified_nameinputs
?
e
F__inference_dropout0.5_layer_call_and_return_conditional_losses_239426

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_bn1_layer_call_fn_239209

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:?????????.?@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2378452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????.?@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????.?@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????.?@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
F__inference_dropout0.5_layer_call_and_return_conditional_losses_239431

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_238652	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*+
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_2374642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????0?

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_bn2_layer_call_fn_239296

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:?????????C?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2379352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????C?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????C?::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????C?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_bn2_layer_call_and_return_conditional_losses_239358

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????:::::j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_bn1_layer_call_and_return_conditional_losses_239121

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
?__inference_bn2_layer_call_and_return_conditional_losses_239265

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????C?:?:?:?:?:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:?????????C?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????C?::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:?????????C?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
s
+__inference_prediction_layer_call_fn_239606
inputs_0
inputs_1
inputs_2
inputs_3
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_2382572
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:?????????:?????????:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/3
?
?
$__inference_bn1_layer_call_fn_239222

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:?????????.?@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2378632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????.?@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????.?@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????.?@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_bn2_layer_call_and_return_conditional_losses_239283

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????C?:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????C?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????C?:::::X T
0
_output_shapes
:?????????C?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
E
)__inference_pooling2_layer_call_fn_237784

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling2_layer_call_and_return_conditional_losses_2377782
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
3__inference_THSR_Captcha_Model_layer_call_fn_239041

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*+
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_2385222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_239390

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????
!?:X T
0
_output_shapes
:?????????
!?
 
_user_specified_nameinputs
?
?
B__inference_digit2_layer_call_and_return_conditional_losses_239470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_digit1_layer_call_fn_239460

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit1_layer_call_and_return_conditional_losses_2381482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_THSR_Captcha_Model_layer_call_fn_238992

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*+
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_2384062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_digit2_layer_call_and_return_conditional_losses_238122

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_bn1_layer_call_and_return_conditional_losses_237863

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????.?@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????.?@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????.?@:::::X T
0
_output_shapes
:?????????.?@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?M
?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238336	
input
conv1_238273
conv1_238275

bn1_238278

bn1_238280

bn1_238282

bn1_238284
conv2_238288
conv2_238290

bn2_238293

bn2_238295

bn2_238297

bn2_238299
dense1_238304
dense1_238306
digit4_238310
digit4_238312
digit3_238315
digit3_238317
digit2_238320
digit2_238322
digit1_238325
digit1_238327
identity??bn1/StatefulPartitionedCall?bn2/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?digit1/StatefulPartitionedCall?digit2/StatefulPartitionedCall?digit3/StatefulPartitionedCall?digit4/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*0
_output_shapes
:?????????0?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_2378022
reshape/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1_238273conv1_238275*
Tin
2*
Tout
2*0
_output_shapes
:?????????.?@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2374762
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
bn1_238278
bn1_238280
bn1_238282
bn1_238284*
Tin	
2*
Tout
2*0
_output_shapes
:?????????.?@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2378632
bn1/StatefulPartitionedCall?
pooling1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????E@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling1_layer_call_and_return_conditional_losses_2376182
pooling1/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall!pooling1/PartitionedCall:output:0conv2_238288conv2_238290*
Tin
2*
Tout
2*0
_output_shapes
:?????????C?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2376362
conv2/StatefulPartitionedCall?
bn2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0
bn2_238293
bn2_238295
bn2_238297
bn2_238299*
Tin	
2*
Tout
2*0
_output_shapes
:?????????C?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2379532
bn2/StatefulPartitionedCall?
pooling2/PartitionedCallPartitionedCall$bn2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????
!?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling2_layer_call_and_return_conditional_losses_2377782
pooling2/PartitionedCall?
flatten/PartitionedCallPartitionedCall!pooling2/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2379962
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_238304dense1_238306*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_2380142 
dense1/StatefulPartitionedCall?
dropout0.5/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout0.5_layer_call_and_return_conditional_losses_2380472
dropout0.5/PartitionedCall?
digit4/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit4_238310digit4_238312*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit4_layer_call_and_return_conditional_losses_2380702 
digit4/StatefulPartitionedCall?
digit3/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit3_238315digit3_238317*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit3_layer_call_and_return_conditional_losses_2380962 
digit3/StatefulPartitionedCall?
digit2/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit2_238320digit2_238322*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit2_layer_call_and_return_conditional_losses_2381222 
digit2/StatefulPartitionedCall?
digit1/StatefulPartitionedCallStatefulPartitionedCall#dropout0.5/PartitionedCall:output:0digit1_238325digit1_238327*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit1_layer_call_and_return_conditional_losses_2381482 
digit1/StatefulPartitionedCall?
reshape_digit1/PartitionedCallPartitionedCall'digit1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_2381772 
reshape_digit1/PartitionedCall?
reshape_digit2/PartitionedCallPartitionedCall'digit2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_2381982 
reshape_digit2/PartitionedCall?
reshape_digit3/PartitionedCallPartitionedCall'digit3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_2382192 
reshape_digit3/PartitionedCall?
reshape_digit4/PartitionedCallPartitionedCall'digit4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_2382402 
reshape_digit4/PartitionedCall?
prediction/PartitionedCallPartitionedCall'reshape_digit1/PartitionedCall:output:0'reshape_digit2/PartitionedCall:output:0'reshape_digit3/PartitionedCall:output:0'reshape_digit4/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_prediction_layer_call_and_return_conditional_losses_2382572
prediction/PartitionedCall?
IdentityIdentity#prediction/PartitionedCall:output:0^bn1/StatefulPartitionedCall^bn2/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^digit1/StatefulPartitionedCall^digit2/StatefulPartitionedCall^digit3/StatefulPartitionedCall^digit4/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2:
bn2/StatefulPartitionedCallbn2/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
digit1/StatefulPartitionedCalldigit1/StatefulPartitionedCall2@
digit2/StatefulPartitionedCalldigit2/StatefulPartitionedCall2@
digit3/StatefulPartitionedCalldigit3/StatefulPartitionedCall2@
digit4/StatefulPartitionedCalldigit4/StatefulPartitionedCall:S O
,
_output_shapes
:?????????0?

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_bn2_layer_call_and_return_conditional_losses_237761

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????:::::j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_237996

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????
!?:X T
0
_output_shapes
:?????????
!?
 
_user_specified_nameinputs
?
e
F__inference_dropout0.5_layer_call_and_return_conditional_losses_238042

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
?__inference_bn2_layer_call_and_return_conditional_losses_237730

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_239566

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?	
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238814

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource
bn2_readvariableop_resource!
bn2_readvariableop_1_resource0
,bn2_fusedbatchnormv3_readvariableop_resource2
.bn2_fusedbatchnormv3_readvariableop_1_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%digit4_matmul_readvariableop_resource*
&digit4_biasadd_readvariableop_resource)
%digit3_matmul_readvariableop_resource*
&digit3_biasadd_readvariableop_resource)
%digit2_matmul_readvariableop_resource*
&digit2_biasadd_readvariableop_resource)
%digit1_matmul_readvariableop_resource*
&digit1_biasadd_readvariableop_resource
identity??'bn1/AssignMovingAvg/AssignSubVariableOp?)bn1/AssignMovingAvg_1/AssignSubVariableOp?'bn2/AssignMovingAvg/AssignSubVariableOp?)bn2/AssignMovingAvg_1/AssignSubVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :02
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????0?2
reshape/Reshape?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dreshape/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????.?@*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????.?@2
conv1/BiasAdds

conv1/ReluReluconv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????.?@2

conv1/Relu?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/Relu:activations:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????.?@:@:@:@:@:*
epsilon%??'72
bn1/FusedBatchNormV3[
	bn1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
	bn1/Const?
bn1/AssignMovingAvg/sub/xConst*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
bn1/AssignMovingAvg/sub/x?
bn1/AssignMovingAvg/subSub"bn1/AssignMovingAvg/sub/x:output:0bn1/Const:output:0*
T0*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn1/AssignMovingAvg/sub?
"bn1/AssignMovingAvg/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02$
"bn1/AssignMovingAvg/ReadVariableOp?
bn1/AssignMovingAvg/sub_1Sub*bn1/AssignMovingAvg/ReadVariableOp:value:0!bn1/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
bn1/AssignMovingAvg/sub_1?
bn1/AssignMovingAvg/mulMulbn1/AssignMovingAvg/sub_1:z:0bn1/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
bn1/AssignMovingAvg/mul?
'bn1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,bn1_fusedbatchnormv3_readvariableop_resourcebn1/AssignMovingAvg/mul:z:0#^bn1/AssignMovingAvg/ReadVariableOp$^bn1/FusedBatchNormV3/ReadVariableOp*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'bn1/AssignMovingAvg/AssignSubVariableOp?
bn1/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
bn1/AssignMovingAvg_1/sub/x?
bn1/AssignMovingAvg_1/subSub$bn1/AssignMovingAvg_1/sub/x:output:0bn1/Const:output:0*
T0*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn1/AssignMovingAvg_1/sub?
$bn1/AssignMovingAvg_1/ReadVariableOpReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02&
$bn1/AssignMovingAvg_1/ReadVariableOp?
bn1/AssignMovingAvg_1/sub_1Sub,bn1/AssignMovingAvg_1/ReadVariableOp:value:0%bn1/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
bn1/AssignMovingAvg_1/sub_1?
bn1/AssignMovingAvg_1/mulMulbn1/AssignMovingAvg_1/sub_1:z:0bn1/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
bn1/AssignMovingAvg_1/mul?
)bn1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resourcebn1/AssignMovingAvg_1/mul:z:0%^bn1/AssignMovingAvg_1/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)bn1/AssignMovingAvg_1/AssignSubVariableOp?
pooling1/MaxPoolMaxPoolbn1/FusedBatchNormV3:y:0*/
_output_shapes
:?????????E@*
ksize
*
paddingVALID*
strides
2
pooling1/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dpooling1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????C?*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????C?2
conv2/BiasAdds

conv2/ReluReluconv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????C?2

conv2/Relu?
bn2/ReadVariableOpReadVariableOpbn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn2/ReadVariableOp?
bn2/ReadVariableOp_1ReadVariableOpbn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn2/ReadVariableOp_1?
#bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#bn2/FusedBatchNormV3/ReadVariableOp?
%bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02'
%bn2/FusedBatchNormV3/ReadVariableOp_1?
bn2/FusedBatchNormV3FusedBatchNormV3conv2/Relu:activations:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????C?:?:?:?:?:*
epsilon%??'72
bn2/FusedBatchNormV3[
	bn2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
	bn2/Const?
bn2/AssignMovingAvg/sub/xConst*?
_class5
31loc:@bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
bn2/AssignMovingAvg/sub/x?
bn2/AssignMovingAvg/subSub"bn2/AssignMovingAvg/sub/x:output:0bn2/Const:output:0*
T0*?
_class5
31loc:@bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn2/AssignMovingAvg/sub?
"bn2/AssignMovingAvg/ReadVariableOpReadVariableOp,bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"bn2/AssignMovingAvg/ReadVariableOp?
bn2/AssignMovingAvg/sub_1Sub*bn2/AssignMovingAvg/ReadVariableOp:value:0!bn2/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
bn2/AssignMovingAvg/sub_1?
bn2/AssignMovingAvg/mulMulbn2/AssignMovingAvg/sub_1:z:0bn2/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:?2
bn2/AssignMovingAvg/mul?
'bn2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,bn2_fusedbatchnormv3_readvariableop_resourcebn2/AssignMovingAvg/mul:z:0#^bn2/AssignMovingAvg/ReadVariableOp$^bn2/FusedBatchNormV3/ReadVariableOp*?
_class5
31loc:@bn2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'bn2/AssignMovingAvg/AssignSubVariableOp?
bn2/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
bn2/AssignMovingAvg_1/sub/x?
bn2/AssignMovingAvg_1/subSub$bn2/AssignMovingAvg_1/sub/x:output:0bn2/Const:output:0*
T0*A
_class7
53loc:@bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn2/AssignMovingAvg_1/sub?
$bn2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$bn2/AssignMovingAvg_1/ReadVariableOp?
bn2/AssignMovingAvg_1/sub_1Sub,bn2/AssignMovingAvg_1/ReadVariableOp:value:0%bn2/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
bn2/AssignMovingAvg_1/sub_1?
bn2/AssignMovingAvg_1/mulMulbn2/AssignMovingAvg_1/sub_1:z:0bn2/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:?2
bn2/AssignMovingAvg_1/mul?
)bn2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resourcebn2/AssignMovingAvg_1/mul:z:0%^bn2/AssignMovingAvg_1/ReadVariableOp&^bn2/FusedBatchNormV3/ReadVariableOp_1*A
_class7
53loc:@bn2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)bn2/AssignMovingAvg_1/AssignSubVariableOp?
pooling2/MaxPoolMaxPoolbn2/FusedBatchNormV3:y:0*0
_output_shapes
:?????????
!?*
ksize
*
paddingVALID*
strides
2
pooling2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten/Const?
flatten/ReshapeReshapepooling2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense1/BiasAddy
dropout0.5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout0.5/dropout/Const?
dropout0.5/dropout/MulMuldense1/BiasAdd:output:0!dropout0.5/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout0.5/dropout/Mul{
dropout0.5/dropout/ShapeShapedense1/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout0.5/dropout/Shape?
/dropout0.5/dropout/random_uniform/RandomUniformRandomUniform!dropout0.5/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout0.5/dropout/random_uniform/RandomUniform?
!dropout0.5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout0.5/dropout/GreaterEqual/y?
dropout0.5/dropout/GreaterEqualGreaterEqual8dropout0.5/dropout/random_uniform/RandomUniform:output:0*dropout0.5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout0.5/dropout/GreaterEqual?
dropout0.5/dropout/CastCast#dropout0.5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout0.5/dropout/Cast?
dropout0.5/dropout/Mul_1Muldropout0.5/dropout/Mul:z:0dropout0.5/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout0.5/dropout/Mul_1?
digit4/MatMul/ReadVariableOpReadVariableOp%digit4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit4/MatMul/ReadVariableOp?
digit4/MatMulMatMuldropout0.5/dropout/Mul_1:z:0$digit4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit4/MatMul?
digit4/BiasAdd/ReadVariableOpReadVariableOp&digit4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit4/BiasAdd/ReadVariableOp?
digit4/BiasAddBiasAdddigit4/MatMul:product:0%digit4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit4/BiasAdd?
digit3/MatMul/ReadVariableOpReadVariableOp%digit3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit3/MatMul/ReadVariableOp?
digit3/MatMulMatMuldropout0.5/dropout/Mul_1:z:0$digit3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit3/MatMul?
digit3/BiasAdd/ReadVariableOpReadVariableOp&digit3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit3/BiasAdd/ReadVariableOp?
digit3/BiasAddBiasAdddigit3/MatMul:product:0%digit3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit3/BiasAdd?
digit2/MatMul/ReadVariableOpReadVariableOp%digit2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit2/MatMul/ReadVariableOp?
digit2/MatMulMatMuldropout0.5/dropout/Mul_1:z:0$digit2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit2/MatMul?
digit2/BiasAdd/ReadVariableOpReadVariableOp&digit2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit2/BiasAdd/ReadVariableOp?
digit2/BiasAddBiasAdddigit2/MatMul:product:0%digit2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit2/BiasAdd?
digit1/MatMul/ReadVariableOpReadVariableOp%digit1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit1/MatMul/ReadVariableOp?
digit1/MatMulMatMuldropout0.5/dropout/Mul_1:z:0$digit1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit1/MatMul?
digit1/BiasAdd/ReadVariableOpReadVariableOp&digit1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit1/BiasAdd/ReadVariableOp?
digit1/BiasAddBiasAdddigit1/MatMul:product:0%digit1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit1/BiasAdds
reshape_digit1/ShapeShapedigit1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit1/Shape?
"reshape_digit1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit1/strided_slice/stack?
$reshape_digit1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit1/strided_slice/stack_1?
$reshape_digit1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit1/strided_slice/stack_2?
reshape_digit1/strided_sliceStridedSlicereshape_digit1/Shape:output:0+reshape_digit1/strided_slice/stack:output:0-reshape_digit1/strided_slice/stack_1:output:0-reshape_digit1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit1/strided_slice?
reshape_digit1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit1/Reshape/shape/1?
reshape_digit1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit1/Reshape/shape/2?
reshape_digit1/Reshape/shapePack%reshape_digit1/strided_slice:output:0'reshape_digit1/Reshape/shape/1:output:0'reshape_digit1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit1/Reshape/shape?
reshape_digit1/ReshapeReshapedigit1/BiasAdd:output:0%reshape_digit1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit1/Reshapes
reshape_digit2/ShapeShapedigit2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit2/Shape?
"reshape_digit2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit2/strided_slice/stack?
$reshape_digit2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit2/strided_slice/stack_1?
$reshape_digit2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit2/strided_slice/stack_2?
reshape_digit2/strided_sliceStridedSlicereshape_digit2/Shape:output:0+reshape_digit2/strided_slice/stack:output:0-reshape_digit2/strided_slice/stack_1:output:0-reshape_digit2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit2/strided_slice?
reshape_digit2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit2/Reshape/shape/1?
reshape_digit2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit2/Reshape/shape/2?
reshape_digit2/Reshape/shapePack%reshape_digit2/strided_slice:output:0'reshape_digit2/Reshape/shape/1:output:0'reshape_digit2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit2/Reshape/shape?
reshape_digit2/ReshapeReshapedigit2/BiasAdd:output:0%reshape_digit2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit2/Reshapes
reshape_digit3/ShapeShapedigit3/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit3/Shape?
"reshape_digit3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit3/strided_slice/stack?
$reshape_digit3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit3/strided_slice/stack_1?
$reshape_digit3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit3/strided_slice/stack_2?
reshape_digit3/strided_sliceStridedSlicereshape_digit3/Shape:output:0+reshape_digit3/strided_slice/stack:output:0-reshape_digit3/strided_slice/stack_1:output:0-reshape_digit3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit3/strided_slice?
reshape_digit3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit3/Reshape/shape/1?
reshape_digit3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit3/Reshape/shape/2?
reshape_digit3/Reshape/shapePack%reshape_digit3/strided_slice:output:0'reshape_digit3/Reshape/shape/1:output:0'reshape_digit3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit3/Reshape/shape?
reshape_digit3/ReshapeReshapedigit3/BiasAdd:output:0%reshape_digit3/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit3/Reshapes
reshape_digit4/ShapeShapedigit4/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit4/Shape?
"reshape_digit4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit4/strided_slice/stack?
$reshape_digit4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit4/strided_slice/stack_1?
$reshape_digit4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit4/strided_slice/stack_2?
reshape_digit4/strided_sliceStridedSlicereshape_digit4/Shape:output:0+reshape_digit4/strided_slice/stack:output:0-reshape_digit4/strided_slice/stack_1:output:0-reshape_digit4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit4/strided_slice?
reshape_digit4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit4/Reshape/shape/1?
reshape_digit4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit4/Reshape/shape/2?
reshape_digit4/Reshape/shapePack%reshape_digit4/strided_slice:output:0'reshape_digit4/Reshape/shape/1:output:0'reshape_digit4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit4/Reshape/shape?
reshape_digit4/ReshapeReshapedigit4/BiasAdd:output:0%reshape_digit4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit4/Reshaper
prediction/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
prediction/concat/axis?
prediction/concatConcatV2reshape_digit1/Reshape:output:0reshape_digit2/Reshape:output:0reshape_digit3/Reshape:output:0reshape_digit4/Reshape:output:0prediction/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????2
prediction/concat?
IdentityIdentityprediction/concat:output:0(^bn1/AssignMovingAvg/AssignSubVariableOp*^bn1/AssignMovingAvg_1/AssignSubVariableOp(^bn2/AssignMovingAvg/AssignSubVariableOp*^bn2/AssignMovingAvg_1/AssignSubVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::2R
'bn1/AssignMovingAvg/AssignSubVariableOp'bn1/AssignMovingAvg/AssignSubVariableOp2V
)bn1/AssignMovingAvg_1/AssignSubVariableOp)bn1/AssignMovingAvg_1/AssignSubVariableOp2R
'bn2/AssignMovingAvg/AssignSubVariableOp'bn2/AssignMovingAvg/AssignSubVariableOp2V
)bn2/AssignMovingAvg_1/AssignSubVariableOp)bn2/AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_dense1_layer_call_fn_239414

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_2380142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_238177

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ǌ
?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238943

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource
bn2_readvariableop_resource!
bn2_readvariableop_1_resource0
,bn2_fusedbatchnormv3_readvariableop_resource2
.bn2_fusedbatchnormv3_readvariableop_1_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%digit4_matmul_readvariableop_resource*
&digit4_biasadd_readvariableop_resource)
%digit3_matmul_readvariableop_resource*
&digit3_biasadd_readvariableop_resource)
%digit2_matmul_readvariableop_resource*
&digit2_biasadd_readvariableop_resource)
%digit1_matmul_readvariableop_resource*
&digit1_biasadd_readvariableop_resource
identity?T
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :02
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????0?2
reshape/Reshape?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dreshape/Reshape:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????.?@*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????.?@2
conv1/BiasAdds

conv1/ReluReluconv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????.?@2

conv1/Relu?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/Relu:activations:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????.?@:@:@:@:@:*
epsilon%??'7*
is_training( 2
bn1/FusedBatchNormV3?
pooling1/MaxPoolMaxPoolbn1/FusedBatchNormV3:y:0*/
_output_shapes
:?????????E@*
ksize
*
paddingVALID*
strides
2
pooling1/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dpooling1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????C?*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????C?2
conv2/BiasAdds

conv2/ReluReluconv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????C?2

conv2/Relu?
bn2/ReadVariableOpReadVariableOpbn2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn2/ReadVariableOp?
bn2/ReadVariableOp_1ReadVariableOpbn2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn2/ReadVariableOp_1?
#bn2/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#bn2/FusedBatchNormV3/ReadVariableOp?
%bn2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02'
%bn2/FusedBatchNormV3/ReadVariableOp_1?
bn2/FusedBatchNormV3FusedBatchNormV3conv2/Relu:activations:0bn2/ReadVariableOp:value:0bn2/ReadVariableOp_1:value:0+bn2/FusedBatchNormV3/ReadVariableOp:value:0-bn2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????C?:?:?:?:?:*
epsilon%??'7*
is_training( 2
bn2/FusedBatchNormV3?
pooling2/MaxPoolMaxPoolbn2/FusedBatchNormV3:y:0*0
_output_shapes
:?????????
!?*
ksize
*
paddingVALID*
strides
2
pooling2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten/Const?
flatten/ReshapeReshapepooling2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense1/BiasAdd?
dropout0.5/IdentityIdentitydense1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dropout0.5/Identity?
digit4/MatMul/ReadVariableOpReadVariableOp%digit4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit4/MatMul/ReadVariableOp?
digit4/MatMulMatMuldropout0.5/Identity:output:0$digit4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit4/MatMul?
digit4/BiasAdd/ReadVariableOpReadVariableOp&digit4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit4/BiasAdd/ReadVariableOp?
digit4/BiasAddBiasAdddigit4/MatMul:product:0%digit4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit4/BiasAdd?
digit3/MatMul/ReadVariableOpReadVariableOp%digit3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit3/MatMul/ReadVariableOp?
digit3/MatMulMatMuldropout0.5/Identity:output:0$digit3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit3/MatMul?
digit3/BiasAdd/ReadVariableOpReadVariableOp&digit3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit3/BiasAdd/ReadVariableOp?
digit3/BiasAddBiasAdddigit3/MatMul:product:0%digit3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit3/BiasAdd?
digit2/MatMul/ReadVariableOpReadVariableOp%digit2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit2/MatMul/ReadVariableOp?
digit2/MatMulMatMuldropout0.5/Identity:output:0$digit2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit2/MatMul?
digit2/BiasAdd/ReadVariableOpReadVariableOp&digit2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit2/BiasAdd/ReadVariableOp?
digit2/BiasAddBiasAdddigit2/MatMul:product:0%digit2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit2/BiasAdd?
digit1/MatMul/ReadVariableOpReadVariableOp%digit1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
digit1/MatMul/ReadVariableOp?
digit1/MatMulMatMuldropout0.5/Identity:output:0$digit1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit1/MatMul?
digit1/BiasAdd/ReadVariableOpReadVariableOp&digit1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
digit1/BiasAdd/ReadVariableOp?
digit1/BiasAddBiasAdddigit1/MatMul:product:0%digit1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
digit1/BiasAdds
reshape_digit1/ShapeShapedigit1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit1/Shape?
"reshape_digit1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit1/strided_slice/stack?
$reshape_digit1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit1/strided_slice/stack_1?
$reshape_digit1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit1/strided_slice/stack_2?
reshape_digit1/strided_sliceStridedSlicereshape_digit1/Shape:output:0+reshape_digit1/strided_slice/stack:output:0-reshape_digit1/strided_slice/stack_1:output:0-reshape_digit1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit1/strided_slice?
reshape_digit1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit1/Reshape/shape/1?
reshape_digit1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit1/Reshape/shape/2?
reshape_digit1/Reshape/shapePack%reshape_digit1/strided_slice:output:0'reshape_digit1/Reshape/shape/1:output:0'reshape_digit1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit1/Reshape/shape?
reshape_digit1/ReshapeReshapedigit1/BiasAdd:output:0%reshape_digit1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit1/Reshapes
reshape_digit2/ShapeShapedigit2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit2/Shape?
"reshape_digit2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit2/strided_slice/stack?
$reshape_digit2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit2/strided_slice/stack_1?
$reshape_digit2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit2/strided_slice/stack_2?
reshape_digit2/strided_sliceStridedSlicereshape_digit2/Shape:output:0+reshape_digit2/strided_slice/stack:output:0-reshape_digit2/strided_slice/stack_1:output:0-reshape_digit2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit2/strided_slice?
reshape_digit2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit2/Reshape/shape/1?
reshape_digit2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit2/Reshape/shape/2?
reshape_digit2/Reshape/shapePack%reshape_digit2/strided_slice:output:0'reshape_digit2/Reshape/shape/1:output:0'reshape_digit2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit2/Reshape/shape?
reshape_digit2/ReshapeReshapedigit2/BiasAdd:output:0%reshape_digit2/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit2/Reshapes
reshape_digit3/ShapeShapedigit3/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit3/Shape?
"reshape_digit3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit3/strided_slice/stack?
$reshape_digit3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit3/strided_slice/stack_1?
$reshape_digit3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit3/strided_slice/stack_2?
reshape_digit3/strided_sliceStridedSlicereshape_digit3/Shape:output:0+reshape_digit3/strided_slice/stack:output:0-reshape_digit3/strided_slice/stack_1:output:0-reshape_digit3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit3/strided_slice?
reshape_digit3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit3/Reshape/shape/1?
reshape_digit3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit3/Reshape/shape/2?
reshape_digit3/Reshape/shapePack%reshape_digit3/strided_slice:output:0'reshape_digit3/Reshape/shape/1:output:0'reshape_digit3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit3/Reshape/shape?
reshape_digit3/ReshapeReshapedigit3/BiasAdd:output:0%reshape_digit3/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit3/Reshapes
reshape_digit4/ShapeShapedigit4/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_digit4/Shape?
"reshape_digit4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"reshape_digit4/strided_slice/stack?
$reshape_digit4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit4/strided_slice/stack_1?
$reshape_digit4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$reshape_digit4/strided_slice/stack_2?
reshape_digit4/strided_sliceStridedSlicereshape_digit4/Shape:output:0+reshape_digit4/strided_slice/stack:output:0-reshape_digit4/strided_slice/stack_1:output:0-reshape_digit4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_digit4/strided_slice?
reshape_digit4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit4/Reshape/shape/1?
reshape_digit4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
reshape_digit4/Reshape/shape/2?
reshape_digit4/Reshape/shapePack%reshape_digit4/strided_slice:output:0'reshape_digit4/Reshape/shape/1:output:0'reshape_digit4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_digit4/Reshape/shape?
reshape_digit4/ReshapeReshapedigit4/BiasAdd:output:0%reshape_digit4/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape_digit4/Reshaper
prediction/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
prediction/concat/axis?
prediction/concatConcatV2reshape_digit1/Reshape:output:0reshape_digit2/Reshape:output:0reshape_digit3/Reshape:output:0reshape_digit4/Reshape:output:0prediction/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????2
prediction/concatr
IdentityIdentityprediction/concat:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?:::::::::::::::::::::::T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
K
/__inference_reshape_digit3_layer_call_fn_239571

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_2382192
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_239055

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :02
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????0?2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:?????????0?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????0?:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs
?
|
'__inference_digit4_layer_call_fn_239517

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit4_layer_call_and_return_conditional_losses_2380702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
?__inference_bn2_layer_call_and_return_conditional_losses_237953

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????C?:?:?:?:?:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????C?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????C?:::::X T
0
_output_shapes
:?????????C?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_238198

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_digit4_layer_call_and_return_conditional_losses_238070

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense1_layer_call_and_return_conditional_losses_239405

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:::Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_digit2_layer_call_fn_239479

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit2_layer_call_and_return_conditional_losses_2381222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_238219

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_pooling1_layer_call_and_return_conditional_losses_237618

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_pooling1_layer_call_fn_237624

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_pooling1_layer_call_and_return_conditional_losses_2376182
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_237802

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :02
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????0?2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:?????????0?2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????0?:T P
,
_output_shapes
:?????????0?
 
_user_specified_nameinputs
?
?
?__inference_bn1_layer_call_and_return_conditional_losses_237601

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity?t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'7*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
F__inference_dropout0.5_layer_call_and_return_conditional_losses_238047

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_conv1_layer_call_and_return_conditional_losses_237476

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
K
/__inference_reshape_digit4_layer_call_fn_239589

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_2382402
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
?__inference_bn1_layer_call_and_return_conditional_losses_239103

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense1_layer_call_and_return_conditional_losses_238014

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:::Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_digit3_layer_call_fn_239498

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_digit3_layer_call_and_return_conditional_losses_2380962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
A__inference_conv2_layer_call_and_return_conditional_losses_237636

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_bn1_layer_call_fn_239147

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2376012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_bn2_layer_call_fn_239371

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2377302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_THSR_Captcha_Model_layer_call_fn_238569	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*+
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_2385222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesr
p:?????????0?::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:?????????0?

_user_specified_nameinput:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_prediction_layer_call_and_return_conditional_losses_239598
inputs_0
inputs_1
inputs_2
inputs_3
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:?????????:?????????:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/3
?
?
B__inference_digit3_layer_call_and_return_conditional_losses_239489

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?$
?
?__inference_bn1_layer_call_and_return_conditional_losses_237570

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%??'72
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
Const?
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg/sub/x?
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1?
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  ??2
AssignMovingAvg_1/sub/x?
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_conv1_layer_call_fn_237486

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2374762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_bn2_layer_call_fn_239309

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:?????????C?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2379532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????C?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????C?::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????C?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_bn2_layer_call_fn_239384

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn2_layer_call_and_return_conditional_losses_2377612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_digit4_layer_call_and_return_conditional_losses_239508

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
K
/__inference_reshape_digit2_layer_call_fn_239553

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_2381982
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_239584

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_pooling2_layer_call_and_return_conditional_losses_237778

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_bn1_layer_call_fn_239134

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_bn1_layer_call_and_return_conditional_losses_2375702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input3
serving_default_input:0?????????0?B

prediction4
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
	optimizer
_layers
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"҄
_tf_keras_model??{"class_name": "Model", "name": "THSR_Captcha_Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "THSR_Captcha_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 140]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [48, 140, 1]}}, "name": "reshape", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pooling1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pooling1", "inbound_nodes": [[["bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["pooling1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pooling2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pooling2", "inbound_nodes": [[["bn2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["pooling2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout0.5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout0.5", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit1", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit1", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit2", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit2", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit3", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit3", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit4", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit4", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit1", "inbound_nodes": [[["digit1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit2", "inbound_nodes": [[["digit2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit3", "inbound_nodes": [[["digit3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit4", "inbound_nodes": [[["digit4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "prediction", "trainable": true, "dtype": "float32", "axis": 1}, "name": "prediction", "inbound_nodes": [[["reshape_digit1", 0, 0, {}], ["reshape_digit2", 0, 0, {}], ["reshape_digit3", 0, 0, {}], ["reshape_digit4", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["prediction", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 140]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "THSR_Captcha_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 140]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [48, 140, 1]}}, "name": "reshape", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pooling1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pooling1", "inbound_nodes": [[["bn1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["pooling1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pooling2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pooling2", "inbound_nodes": [[["bn2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["pooling2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout0.5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout0.5", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit1", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit1", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit2", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit2", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit3", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit3", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "digit4", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "digit4", "inbound_nodes": [[["dropout0.5", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit1", "inbound_nodes": [[["digit1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit2", "inbound_nodes": [[["digit2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit3", "inbound_nodes": [[["digit3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_digit4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}, "name": "reshape_digit4", "inbound_nodes": [[["digit4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "prediction", "trainable": true, "dtype": "float32", "axis": 1}, "name": "prediction", "inbound_nodes": [[["reshape_digit1", 0, 0, {}], ["reshape_digit2", 0, 0, {}], ["reshape_digit3", 0, 0, {}], ["reshape_digit4", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["prediction", 0, 0]]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": [{"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}}, {"class_name": "AllDigitsAccuracy", "config": {"name": "all_digits_acc", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 140]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 140]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [48, 140, 1]}}}
?	

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 140, 1]}}
?
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+trainable_variables
,regularization_losses
-	variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 138, 64]}}
?
/trainable_variables
0regularization_losses
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "pooling1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "pooling1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 69, 64]}}
?
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>trainable_variables
?regularization_losses
@	variables
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 1e-05, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 67, 128]}}
?
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "pooling2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "pooling2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 42240}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42240]}}
?
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout0.5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout0.5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Tkernel
Ubias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "digit1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "digit1", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

Zkernel
[bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "digit2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "digit2", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "digit3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "digit3", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "digit4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "digit4", "trainable": true, "dtype": "float32", "units": 19, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_digit1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_digit1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}}
?
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_digit2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_digit2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}}
?
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_digit3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_digit3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}}
?
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_digit4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_digit4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 19]}}}
?
|trainable_variables
}regularization_losses
~	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "prediction", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "prediction", "trainable": true, "dtype": "float32", "axis": 1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 19]}, {"class_name": "TensorShape", "items": [null, 1, 19]}, {"class_name": "TensorShape", "items": [null, 1, 19]}, {"class_name": "TensorShape", "items": [null, 1, 19]}]}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate m?!m?'m?(m?3m?4m?:m?;m?Jm?Km?Tm?Um?Zm?[m?`m?am?fm?gm? v?!v?'v?(v?3v?4v?:v?;v?Jv?Kv?Tv?Uv?Zv?[v?`v?av?fv?gv?"
	optimizer
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
?
 0
!1
'2
(3
34
45
:6
;7
J8
K9
T10
U11
Z12
[13
`14
a15
f16
g17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 0
!1
'2
(3
)4
*5
36
47
:8
;9
<10
=11
J12
K13
T14
U15
Z16
[17
`18
a19
f20
g21"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
trainable_variables
?layer_metrics
regularization_losses
	variables
?metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
trainable_variables
?layer_metrics
regularization_losses
	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@2conv1/kernel
:@2
conv1/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
"trainable_variables
?layer_metrics
#regularization_losses
$	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2	bn1/gamma
:@2bn1/beta
:@ (2bn1/moving_mean
#:!@ (2bn1/moving_variance
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
+trainable_variables
?layer_metrics
,regularization_losses
-	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
/trainable_variables
?layer_metrics
0regularization_losses
1	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@?2conv2/kernel
:?2
conv2/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
5trainable_variables
?layer_metrics
6regularization_losses
7	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2	bn2/gamma
:?2bn2/beta
 :? (2bn2/moving_mean
$:"? (2bn2/moving_variance
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
:0
;1
<2
=3"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
>trainable_variables
?layer_metrics
?regularization_losses
@	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Btrainable_variables
?layer_metrics
Cregularization_losses
D	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Ftrainable_variables
?layer_metrics
Gregularization_losses
H	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": ???2dense1/kernel
:?2dense1/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Ltrainable_variables
?layer_metrics
Mregularization_losses
N	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Ptrainable_variables
?layer_metrics
Qregularization_losses
R	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2digit1/kernel
:2digit1/bias
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
Vtrainable_variables
?layer_metrics
Wregularization_losses
X	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2digit2/kernel
:2digit2/bias
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
\trainable_variables
?layer_metrics
]regularization_losses
^	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2digit3/kernel
:2digit3/bias
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
btrainable_variables
?layer_metrics
cregularization_losses
d	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2digit4/kernel
:2digit4/bias
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
htrainable_variables
?layer_metrics
iregularization_losses
j	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
ltrainable_variables
?layer_metrics
mregularization_losses
n	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
ptrainable_variables
?layer_metrics
qregularization_losses
r	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
ttrainable_variables
?layer_metrics
uregularization_losses
v	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
xtrainable_variables
?layer_metrics
yregularization_losses
z	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layers
|trainable_variables
?layer_metrics
}regularization_losses
~	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
)0
*1
<2
=3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_dict_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "CategoricalAccuracy", "name": "categorical_accuracy", "dtype": "float32", "config": {"name": "categorical_accuracy", "dtype": "float32"}}
?

?adacc
?ad_acc
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "AllDigitsAccuracy", "name": "all_digits_acc", "dtype": "float32", "config": {"name": "all_digits_acc", "dtype": "float32"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2adacc
(
?0"
trackable_list_wrapper
.
?	variables"
_generic_user_object
+:)@2Adam/conv1/kernel/m
:@2Adam/conv1/bias/m
:@2Adam/bn1/gamma/m
:@2Adam/bn1/beta/m
,:*@?2Adam/conv2/kernel/m
:?2Adam/conv2/bias/m
:?2Adam/bn2/gamma/m
:?2Adam/bn2/beta/m
':%???2Adam/dense1/kernel/m
:?2Adam/dense1/bias/m
%:#	?2Adam/digit1/kernel/m
:2Adam/digit1/bias/m
%:#	?2Adam/digit2/kernel/m
:2Adam/digit2/bias/m
%:#	?2Adam/digit3/kernel/m
:2Adam/digit3/bias/m
%:#	?2Adam/digit4/kernel/m
:2Adam/digit4/bias/m
+:)@2Adam/conv1/kernel/v
:@2Adam/conv1/bias/v
:@2Adam/bn1/gamma/v
:@2Adam/bn1/beta/v
,:*@?2Adam/conv2/kernel/v
:?2Adam/conv2/bias/v
:?2Adam/bn2/gamma/v
:?2Adam/bn2/beta/v
':%???2Adam/dense1/kernel/v
:?2Adam/dense1/bias/v
%:#	?2Adam/digit1/kernel/v
:2Adam/digit1/bias/v
%:#	?2Adam/digit2/kernel/v
:2Adam/digit2/bias/v
%:#	?2Adam/digit3/kernel/v
:2Adam/digit3/bias/v
%:#	?2Adam/digit4/kernel/v
:2Adam/digit4/bias/v
?2?
3__inference_THSR_Captcha_Model_layer_call_fn_238453
3__inference_THSR_Captcha_Model_layer_call_fn_238992
3__inference_THSR_Captcha_Model_layer_call_fn_238569
3__inference_THSR_Captcha_Model_layer_call_fn_239041?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238269
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238943
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238814
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238336?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_237464?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *)?&
$?!
input?????????0?
?2?
(__inference_reshape_layer_call_fn_239060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_reshape_layer_call_and_return_conditional_losses_239055?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv1_layer_call_fn_237486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
A__inference_conv1_layer_call_and_return_conditional_losses_237476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
$__inference_bn1_layer_call_fn_239134
$__inference_bn1_layer_call_fn_239209
$__inference_bn1_layer_call_fn_239147
$__inference_bn1_layer_call_fn_239222?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_bn1_layer_call_and_return_conditional_losses_239121
?__inference_bn1_layer_call_and_return_conditional_losses_239103
?__inference_bn1_layer_call_and_return_conditional_losses_239196
?__inference_bn1_layer_call_and_return_conditional_losses_239178?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_pooling1_layer_call_fn_237624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_pooling1_layer_call_and_return_conditional_losses_237618?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_conv2_layer_call_fn_237646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
A__inference_conv2_layer_call_and_return_conditional_losses_237636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
$__inference_bn2_layer_call_fn_239309
$__inference_bn2_layer_call_fn_239371
$__inference_bn2_layer_call_fn_239384
$__inference_bn2_layer_call_fn_239296?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_bn2_layer_call_and_return_conditional_losses_239265
?__inference_bn2_layer_call_and_return_conditional_losses_239358
?__inference_bn2_layer_call_and_return_conditional_losses_239340
?__inference_bn2_layer_call_and_return_conditional_losses_239283?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_pooling2_layer_call_fn_237784?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_pooling2_layer_call_and_return_conditional_losses_237778?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_flatten_layer_call_fn_239395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_239390?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense1_layer_call_fn_239414?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense1_layer_call_and_return_conditional_losses_239405?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout0.5_layer_call_fn_239441
+__inference_dropout0.5_layer_call_fn_239436?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout0.5_layer_call_and_return_conditional_losses_239431
F__inference_dropout0.5_layer_call_and_return_conditional_losses_239426?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_digit1_layer_call_fn_239460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_digit1_layer_call_and_return_conditional_losses_239451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_digit2_layer_call_fn_239479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_digit2_layer_call_and_return_conditional_losses_239470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_digit3_layer_call_fn_239498?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_digit3_layer_call_and_return_conditional_losses_239489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_digit4_layer_call_fn_239517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_digit4_layer_call_and_return_conditional_losses_239508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_reshape_digit1_layer_call_fn_239535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_239530?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_reshape_digit2_layer_call_fn_239553?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_239548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_reshape_digit3_layer_call_fn_239571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_239566?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_reshape_digit4_layer_call_fn_239589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_239584?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_prediction_layer_call_fn_239606?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_prediction_layer_call_and_return_conditional_losses_239598?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
1B/
$__inference_signature_wrapper_238652input?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238269? !'()*34:;<=JKfg`aZ[TU;?8
1?.
$?!
input?????????0?
p

 
? ")?&
?
0?????????
? ?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238336? !'()*34:;<=JKfg`aZ[TU;?8
1?.
$?!
input?????????0?
p 

 
? ")?&
?
0?????????
? ?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238814? !'()*34:;<=JKfg`aZ[TU<?9
2?/
%?"
inputs?????????0?
p

 
? ")?&
?
0?????????
? ?
N__inference_THSR_Captcha_Model_layer_call_and_return_conditional_losses_238943? !'()*34:;<=JKfg`aZ[TU<?9
2?/
%?"
inputs?????????0?
p 

 
? ")?&
?
0?????????
? ?
3__inference_THSR_Captcha_Model_layer_call_fn_238453s !'()*34:;<=JKfg`aZ[TU;?8
1?.
$?!
input?????????0?
p

 
? "???????????
3__inference_THSR_Captcha_Model_layer_call_fn_238569s !'()*34:;<=JKfg`aZ[TU;?8
1?.
$?!
input?????????0?
p 

 
? "???????????
3__inference_THSR_Captcha_Model_layer_call_fn_238992t !'()*34:;<=JKfg`aZ[TU<?9
2?/
%?"
inputs?????????0?
p

 
? "???????????
3__inference_THSR_Captcha_Model_layer_call_fn_239041t !'()*34:;<=JKfg`aZ[TU<?9
2?/
%?"
inputs?????????0?
p 

 
? "???????????
!__inference__wrapped_model_237464? !'()*34:;<=JKfg`aZ[TU3?0
)?&
$?!
input?????????0?
? ";?8
6

prediction(?%

prediction??????????
?__inference_bn1_layer_call_and_return_conditional_losses_239103?'()*M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
?__inference_bn1_layer_call_and_return_conditional_losses_239121?'()*M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
?__inference_bn1_layer_call_and_return_conditional_losses_239178t'()*<?9
2?/
)?&
inputs?????????.?@
p
? ".?+
$?!
0?????????.?@
? ?
?__inference_bn1_layer_call_and_return_conditional_losses_239196t'()*<?9
2?/
)?&
inputs?????????.?@
p 
? ".?+
$?!
0?????????.?@
? ?
$__inference_bn1_layer_call_fn_239134?'()*M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
$__inference_bn1_layer_call_fn_239147?'()*M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
$__inference_bn1_layer_call_fn_239209g'()*<?9
2?/
)?&
inputs?????????.?@
p
? "!??????????.?@?
$__inference_bn1_layer_call_fn_239222g'()*<?9
2?/
)?&
inputs?????????.?@
p 
? "!??????????.?@?
?__inference_bn2_layer_call_and_return_conditional_losses_239265t:;<=<?9
2?/
)?&
inputs?????????C?
p
? ".?+
$?!
0?????????C?
? ?
?__inference_bn2_layer_call_and_return_conditional_losses_239283t:;<=<?9
2?/
)?&
inputs?????????C?
p 
? ".?+
$?!
0?????????C?
? ?
?__inference_bn2_layer_call_and_return_conditional_losses_239340?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
?__inference_bn2_layer_call_and_return_conditional_losses_239358?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
$__inference_bn2_layer_call_fn_239296g:;<=<?9
2?/
)?&
inputs?????????C?
p
? "!??????????C??
$__inference_bn2_layer_call_fn_239309g:;<=<?9
2?/
)?&
inputs?????????C?
p 
? "!??????????C??
$__inference_bn2_layer_call_fn_239371?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
$__inference_bn2_layer_call_fn_239384?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
A__inference_conv1_layer_call_and_return_conditional_losses_237476? !I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
&__inference_conv1_layer_call_fn_237486? !I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????@?
A__inference_conv2_layer_call_and_return_conditional_losses_237636?34I?F
??<
:?7
inputs+???????????????????????????@
? "@?=
6?3
0,????????????????????????????
? ?
&__inference_conv2_layer_call_fn_237646?34I?F
??<
:?7
inputs+???????????????????????????@
? "3?0,?????????????????????????????
B__inference_dense1_layer_call_and_return_conditional_losses_239405_JK1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? }
'__inference_dense1_layer_call_fn_239414RJK1?.
'?$
"?
inputs???????????
? "????????????
B__inference_digit1_layer_call_and_return_conditional_losses_239451]TU0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_digit1_layer_call_fn_239460PTU0?-
&?#
!?
inputs??????????
? "???????????
B__inference_digit2_layer_call_and_return_conditional_losses_239470]Z[0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_digit2_layer_call_fn_239479PZ[0?-
&?#
!?
inputs??????????
? "???????????
B__inference_digit3_layer_call_and_return_conditional_losses_239489]`a0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_digit3_layer_call_fn_239498P`a0?-
&?#
!?
inputs??????????
? "???????????
B__inference_digit4_layer_call_and_return_conditional_losses_239508]fg0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_digit4_layer_call_fn_239517Pfg0?-
&?#
!?
inputs??????????
? "???????????
F__inference_dropout0.5_layer_call_and_return_conditional_losses_239426^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
F__inference_dropout0.5_layer_call_and_return_conditional_losses_239431^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
+__inference_dropout0.5_layer_call_fn_239436Q4?1
*?'
!?
inputs??????????
p
? "????????????
+__inference_dropout0.5_layer_call_fn_239441Q4?1
*?'
!?
inputs??????????
p 
? "????????????
C__inference_flatten_layer_call_and_return_conditional_losses_239390c8?5
.?+
)?&
inputs?????????
!?
? "'?$
?
0???????????
? ?
(__inference_flatten_layer_call_fn_239395V8?5
.?+
)?&
inputs?????????
!?
? "?????????????
D__inference_pooling1_layer_call_and_return_conditional_losses_237618?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_pooling1_layer_call_fn_237624?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_pooling2_layer_call_and_return_conditional_losses_237778?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
)__inference_pooling2_layer_call_fn_237784?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_prediction_layer_call_and_return_conditional_losses_239598????
???
???
&?#
inputs/0?????????
&?#
inputs/1?????????
&?#
inputs/2?????????
&?#
inputs/3?????????
? ")?&
?
0?????????
? ?
+__inference_prediction_layer_call_fn_239606????
???
???
&?#
inputs/0?????????
&?#
inputs/1?????????
&?#
inputs/2?????????
&?#
inputs/3?????????
? "???????????
J__inference_reshape_digit1_layer_call_and_return_conditional_losses_239530\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
/__inference_reshape_digit1_layer_call_fn_239535O/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_reshape_digit2_layer_call_and_return_conditional_losses_239548\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
/__inference_reshape_digit2_layer_call_fn_239553O/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_reshape_digit3_layer_call_and_return_conditional_losses_239566\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
/__inference_reshape_digit3_layer_call_fn_239571O/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_reshape_digit4_layer_call_and_return_conditional_losses_239584\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
/__inference_reshape_digit4_layer_call_fn_239589O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_reshape_layer_call_and_return_conditional_losses_239055f4?1
*?'
%?"
inputs?????????0?
? ".?+
$?!
0?????????0?
? ?
(__inference_reshape_layer_call_fn_239060Y4?1
*?'
%?"
inputs?????????0?
? "!??????????0??
$__inference_signature_wrapper_238652? !'()*34:;<=JKfg`aZ[TU<?9
? 
2?/
-
input$?!
input?????????0?";?8
6

prediction(?%

prediction?????????