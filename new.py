import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import dataextract as dx
X,Y,X_test,Y_test,ss =dx.datax()
X=X.reshape(-1,200,200,1)
X_test=X_test.reshape([-1,200,200,1])

convnet=input_data(shape=[None,200,200,1],name='input')

convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)

convnet=fully_connected(convnet,5,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='targets')

model=tflearn.DNN(convnet,tensorboard_verbose=0)
model.fit({'input': X}, {'targets': Y}, n_epoch=8,  validation_set=({'input': X_test}, {'targets': Y_test}),
    snapshot_step=500,
    show_metric=True,
    run_id='mnist')
model.save("GestureRecogModel.tfl")
