# ----------------------------------------------------------------------------------------------------------------------
import CNN_Faster_TF_Hub
# ----------------------------------------------------------------------------------------------------------------------
def example_detection():

    #filename_in = 'data/ex-natural/dog/dog_0000.jpg'
    filename_in = 'data/ex13/input.jpg'
    CNN = CNN_Faster_TF_Hub.CNN_Faster_TF()
    CNN.detect_object(filename_in)
    return

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_detection()