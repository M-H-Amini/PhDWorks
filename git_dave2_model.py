#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                  Title: Github Model                                ##
##                                   Date: 2023/05/13                                  ##
##                                                                                     ##
#########################################################################################

##  Description: Load the pretrained model from github


from mh_dave2_model import generateModel
import tensorflow as tf
import logging as log
import os

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##  GPU...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    log.info(f'GPUs: {gpus}')
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    except RuntimeError as e:
        log.error(e)

def loadModel(weights_path):
    model = generateModel((66,200,3))
    model.load_weights(weights_path)
    model.compile(loss='mae', optimizer='adam')
    return model

if __name__ == '__main__':
    weights_path = os.path.join('DAVE2-Keras-master', 'model.h5')
    model = loadModel(weights_path)
    model.summary()
    log.info('Model loaded successfully')


