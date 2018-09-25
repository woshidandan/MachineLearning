  残差网络作为机器学习领域中的重要一部分，在深层次的网络中，发挥了很重要的作用，对于整个网络起到了优化的作用，也为更深层次的神经网络的实现提供了可能。
  (1)主要的公式：
  yl = h(xl) + F(xl;Wl); 
  xl+1 = f(yl): 
  
   ![]( https://github.com/woshidandan/MachineLearning/blob/master/images/residual%20learning.png)  
   
Here xl is the input feature to the l-th Residual Unit. Wl = fWl;kj1kKg is a set of weights (and biases) associated with the l-th Residual Unit, and K is the number of layers in a Residual Unit (K is 2 or 3 in [1]). F denotes the residual function, e.g., a stack of two 33 convolutional layers in [1]. The function f is the operation after element-wise addition, and in [1] f is ReLU. The function h
is set as an identity mapping: h(xl) = xl.1.
  其中，F(x)为残差函数，H(x)为该单元的输出结果，x作为一开始的输入，其来自上层神经网络的综合结果。
  
  (2)目的：
  首先，残差网络并不是通过shortcut connections跳过中间的神经网络层，而是我们通过forward学习中，通过backward的过程，将会促使F(x)中的权重趋向于0，而趋向于0的效果，即是我们等于跳过了这几层，也就是达到了恒等映射，也即跳过的过程。
  With the residual learning reformulation,if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.
  当然，在这个过程中，我们需要其完成恒等映射，所谓的恒等映射，也即：h(xl) = xl，当恒等映射被训练到最优的情况时，参数便趋向于0。
  The function h is set as an identity mapping: h(xl) = xl.
  The central idea of ResNets is to learn the additive residual function F with respect to h(xl), with a key choice of using an identity mapping h(xl) = xl.This is realized by attaching an identity skip connection (\shortcut").
  另一种理解是：当训练的过程中，在到达数据X时，便已经将参数训练到最优的情况，那么无论增加多少层恒等映射，F(x)的部分的参数一定是为0的。
  
  (3)问题：
  
  ![]( https://github.com/woshidandan/MachineLearning/blob/master/images/identityblock.png) 
  
  既然我们已经有了shortcut path，为何还需要main path?也就是添加shortcut path的必要性。
  残差网络的目的是使神经网络更利于优化，而这种优化会体现在：
  (1)若F是求和前网络映射，H是从输入到求和后的网络映射。比如把5映射到5.1，那么引入残差前是F'(5)=5.1，引入残差后是H(5)=5.1, H(5)=F(5)+5, F(5)=0.1。这里的F'和F都表示网络参数映射，引入残差后的映射对输出的变化更敏感。比如s输出从5.1变到5.2，映射F'的输出增加了1/51=2%，而对于残差结构输出从5.1到5.2，映射F是从0.1到0.2，增加了100%。明显后者输出变化对权重的调整作用更大，所以效果更好。残差的思想都是去掉相同的主体部分，从而突出微小的变化。
  (2)现在我们要训练一个深层的网络，它可能过深，假设存在一个性能最强的完美网络N，与它相比我们的网络中必定有一些层是多余的，那么这些多余的层的训练目标是恒等变换，只有达到这个目标我们的网络性能才能跟N一样。对于这些需要实现恒等变换的多余的层，要拟合的目标就成了H(x)=x，在传统网络中，网络的输出目标是F(x)=x，也就是想一步到位，这比较困难，而在残差网络中，拟合的目标成了x-x=0，网络的输出目标为F(x)=0，也就是分很多步骤，逐步来，这比前者要容易得多。
  If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping, than to learn the function as a new one.
  
  (4)残差网络解决的主要问题：
  在深层次的神经网络中，当学习到的内容进行反向传递以修改参数时，面临的问题是梯度消失或者梯度爆炸，而残差网络便很好的解决了这一点：
  
  ![]( https://github.com/woshidandan/MachineLearning/blob/master/images/solveproblem.png) 
  
  ![]( https://github.com/woshidandan/MachineLearning/blob/master/images/solveproblem1.png) 
  
  通过在反向求偏导数的过程中，将原来的累乘换成累加，从而引入常数项，避免了梯度的消失或者梯度的爆炸。
  
  (5)如何构建残差网络:
  1、首先我们需要构建identity block模块
  
   ![]( https://github.com/woshidandan/MachineLearning/blob/master/images/identityblock.png) 
   
   代码如下：
   ```python
   
   def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    print("finished")
    return X
    
   ```
   在代码中，除了正常的卷积操作，以及激活函数的使用，还加入了BatchNormalization方法，这个方法主要是为了加速训练(BN就是在神经网络的训练过程中对每层的输入数据加一个标准化处理)
   
     2、其次我们需要构建convolutional block模块
  
   ![]( https://github.com/woshidandan/MachineLearning/blob/master/images/convolutionalblock.png) 
   
   代码如下:
   
   ```python
   
   def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1, 1), name = conv_name_base + '2b',padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), name = conv_name_base + '1',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
   
   ```
   
   与上面identityblock所不同的是，在shortcut poth中，多了一层对输入数据进行池化和BN的操作，这主要是为了是维度保持一致。
   The CONV2D layer in the shortcut path is used to resize the input  x  to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. (This plays a similar role as the matrix  Ws  discussed in lecture.)
   
   3、完成了必要模块的搭建，我们需要将其引用到实际的残差网络模型中来(50层)
   
   ![]( https://github.com/woshidandan/MachineLearning/blob/master/images/resnet50layers.png) 
   
   代码部分：
   ```python
   
   def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    # The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    # The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
    X = convolutional_block(X, f = 3, filters=[128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='b')
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='c')
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='d')

    # Stage 4 (≈6 lines)
    # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
    X = convolutional_block(X, f = 3, filters=[256, 256, 1024], block='a', stage=4, s = 2)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='b', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='c', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='d', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='e', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='f', stage=4)

    # Stage 5 (≈3 lines)
    # The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    # The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".
    X = convolutional_block(X, f = 3, filters=[512, 512, 2048], stage=5, block='a', s = 2)
    
    # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading
    X = identity_block(X, f = 3, filters=[256, 256, 2048], stage=5, block='b')
    X = identity_block(X, f = 3, filters=[256, 256, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    # The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
    X = AveragePooling2D(pool_size=(2,2))(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
    
   ```
   
   然后是对模型传参和优化：
   ```python
   
   model = ResNet50(input_shape = (64, 64, 3), classes = 6)
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   
   ```
   其中，model.compile使用的是keras中的优化器，即自定义损失函数。
   
   之后是对数据进行传入和加工：
   ```python
      
   X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()resnets_utils.p

  # Normalize image vectors
  X_train = X_train_orig/255.
  X_test = X_test_orig/255.

  # Convert training and test labels to one hot matrices
  Y_train = convert_to_one_hot(Y_train_orig, 6).T
  Y_test = convert_to_one_hot(Y_test_orig, 6).T
  
   ```
   在这个过程中，使用的load_dataset()调用自代码中的resnets_utils.py文件，由此也可以看到数据的来源出
    ```python
    
    def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    ```
     
     之后，通过
     ```python
     
     model.fit(X_train, Y_train, epochs = 20, batch_size = 32)
     
     ```
     其中epochs = 20为训练次数，每次里面会包括很多组
     
     之后是对模型在训练集上表现的评估
  ```python
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
   
  ```
  
    然后，引入另一个数据集ResNet50.h5，对模型的训练结果进行评估
    
    ```python
    model = load_model('ResNet50.h5') 
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    ```
    
    最后，对自己引入的一张照片进行预测
    
    ```python
    img_path = 'images/my_image.jpg'
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    my_image = scipy.misc.imread(img_path)
    imshow(my_image)
    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(model.predict(x))
    
    ```
  
  
     

  
 
 
