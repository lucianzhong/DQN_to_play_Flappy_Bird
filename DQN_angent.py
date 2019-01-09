#!/usr/bin/env python

#Python 2.7 or 3
#TensorFlow 0.7
#pygame
#OpenCV-Python

from __future__ import print_function

import tensorflow as tf
import cv2
import sys
#sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' 				# the name of the game being played for log files
ACTIONS = 2 				# number of valid actions
GAMMA = 0.99 				# decay rate of past observations
OBSERVE = 100000. 			# timesteps to observe before training
EXPLORE = 2000000. 		    # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 		# final value of epsilon
INITIAL_EPSILON = 0.0001    # starting value of epsilon
REPLAY_MEMORY = 50000 		# number of previous transitions to remember
BATCH = 32 					# size of minibatch
FRAME_PER_ACTION = 1

# 权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01) #tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
    return tf.Variable(initial)
#
def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
# 卷积函数
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME") 
    #实现卷积的函数
    #input：输入图片，格式为[batch，长，宽，通道数]，长和宽比较好理解，batch就是一批训练数据有多少张照片，通道数实际上是输入图片的三维矩阵的深度，如果是普通灰度照片，通道数就是1，如果是RGB彩色照片，通道数就是3，当然这个通道数完全可以自己设计。
    #filter：就是卷积核，其格式为[长，宽，输入通道数，输出通道数]，其中长和宽指的是本次卷积计算的“抹布”的规格，输入通道数应当和input的通道数一致，输出通道数可以随意指定。
    #strides:是步长，一般情况下的格式为[1，长上步长，宽上步长，1]，所谓步长就是指抹布（卷积核）每次在长和宽上滑动多少会停下来计算一次卷积。这个步长不一定要能够被输入图片的长和宽整除。
    #padding：是卷积核（抹布）在边缘处的处理方法
# 池化 核 2*2 步长2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    #tf.nn.max_pool(value, ksize, strides, padding, name=None)
    #value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    #ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    #strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    #padding：和卷积类似，可以取'VALID' 或者'SAME'
# CNN
def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])    # 卷积核patch的大小是8*8, RGBD,channel是4,输出是32个featuremap
    b_conv1 = bias_variable([32])				# 传入它的shape为[32]

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer 输入层 输入向量为80*80*4
    s = tf.placeholder("float", [None, 80, 80, 4])					# 
    print("s.shape",s.shape)
    # hidden layers  
    # 第一个隐藏层+一个池化层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)			#  
    h_pool1 = max_pool_2x2(h_conv1)									# 
    print("h_pool1.shape",h_pool1.shape)
    #第二个隐藏层
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)		#   
    # 第三个隐藏层
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)		# 
    print("h_conv3.shape",h_conv3.shape)
    #展平
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    print("h_conv3_flat.shape",h_conv3_flat.shape)
    # 第一个全连接层
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    print("h_fc1.shape",h_fc1.shape)
    # readout layer  输出层
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    print("readout.size",readout.shape)

    return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function  定义损失函数
    a = tf.placeholder("float", [None, ACTIONS]) #tf.placeholder 是 Tensorflow 中的占位符，暂时储存变量
    y = tf.placeholder("float", [None])
    #multiply这个函数实现的是元素级别的相乘
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1) #矩阵按行求和

    cost = tf.reduce_mean( tf.square(y - readout_action) ) #张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值

    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    #将图像转换成80*80，并进行灰度化
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)  #Resize image to 80x80, Convert image to grayscale,remove the background appeared in the original game can make it converge faster
    
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)  #对图像进行二值化,从灰度图像中获取二进制图像或用于消除噪声，即滤除太小或太小的像素
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 将图像处理成4通道,stack last 4 frames to produce an 80x80x4 input array for network

    # saving and loading networks
    saver = tf.train.Saver()													#  将训练好的模型参数保存起来，以便以后进行验证或测试,创建一个Saver对象
    sess.run(tf.initialize_all_variables()) 
    checkpoint = tf.train.get_checkpoint_state("saved_networks")				# checkpoint文件找到模型文件名
    print("checkpoint",checkpoint)    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]   #将当前环境输入到CNN网络中
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)					# 执行选择的动作，并保存返回的状态、得分

        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)

        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        #经验池保存的是以一个马尔科夫序列于D中
        D.append((s_t, a_t, r_t, s_t1, terminal))
        
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:  #OBSERVE = 100000.# timesteps to observe before training
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)  #32

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            y_batch = []  #y_batch表示标签值，如果下一时刻游戏关闭则直接用奖励做标签值，若游戏没有关闭，则要在奖励的基础上加上GAMMA比例的下一时刻最大的模型预测值
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
            # perform gradient step
            train_step.run(  feed_dict = {y : y_batch, a : a_batch, s : s_j_batch}  )

        # update the old values
        s_t = s_t1
        t += 1
        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,  "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,  "/ Q_MAX %e" % np.max(readout_t))



def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)



if __name__ == "__main__":
    playGame()
