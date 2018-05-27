import cv2
import numpy as np
import os
import random


def celeb_loader(dir='/home/airscan-razer04/Documents/datasets/img_align_celeba/',
                    randomize=True,
                    batch_size=64,
                    height=64,
                    width=64,
                    split=False,
                    type='train',
                    norm=False):
    list = os.listdir(dir) # dir is your directory path
    number_files = len(list)
    list.sort()

    if split == True:
        test_list = list[9::10]
        train_list = list[:]
        del train_list[9::10]

        if type == 'train':
            list = train_list
        elif type == 'test':
            list = test_list

    #print(np.array(list).reshape((-1,1)))
    #print (len(list))
    while(1):
        if randomize == True:
            random.shuffle(list)
        img_list = list[:]
        #print(np.array(img_list).reshape((-1,1)))

        while img_list:
            img_stack = np.zeros((batch_size, height, width,3),dtype=np.float32)

            for i in range(batch_size):
                #print(len(img_list), len(list))

                filename = dir + img_list.pop(0)
                #print(filename)
                img = cv2.imread(filename)
                #cv2.imshow("image",img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #print(img.shape)
                if img.shape != (64,64,3):
                    #Convert range: 0 to 1
                    img_stack[i, :, :,:] = cv2.resize(img,(height,width))/255
                    # Convert range: -1 to 1
                    if norm == True:
                        img_stack[i, :, :,:] = img_stack[i, :, :,:]*2 - 1
                    #cv2.imshow("image",img_stack[i, :, :,:])
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                    #print(img.shape)
                #print("imgs left",len(img_list))
                if len(img_list)==0:
                    if randomize == True:
                        random.shuffle(list)
                    img_list = list[:]
            yield img_stack, None
            #cv2.imshow("image",a)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

if __name__ == '__main__':
    print("MAIN")
    some_gen = celeb_loader()
    a,b = next(some_gen)
    #print(a[0,:,:,:])
    # for i in range (a.shape[0]):
    #     cv2.imshow("image",a[i])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
