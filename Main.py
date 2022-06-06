import tensorflow as tf
import numpy as np

from tkinter import *
import os
from tkinter import filedialog
import cv2
import time
from matplotlib import pyplot as plt
from tkinter import messagebox






def endprogram():
	print ("\nProgram terminated!")
	sys.exit()









def training():

    import Training as tr

    '''global training_screen
    training_screen = Toplevel(main_screen)
    training_screen.title("Training")
    # login_screen.geometry("400x300")
    training_screen.geometry("600x450+650+150")
    training_screen.minsize(120, 1)
    training_screen.maxsize(1604, 881)
    training_screen.resizable(1, 1)
    # login_screen.title("New Toplevel")

    Label(training_screen, text='Upload Image ', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000", bg="turquoise", width="300", height="2", font=("Calibri", 16)).pack()
    Label(training_screen, text="").pack()
    Label(training_screen, text="").pack()
    Label(training_screen, text="").pack()
    Button(training_screen, text='Upload Image', font=(
        'Verdana', 15), height="2", width="30", command=imgtraining).pack()'''




def imgtraining():
    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)
    filename = 'Test.jpg'
    cv2.imwrite(filename, image)
    print("After saving image:")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)
    # import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))

    from PIL import Image, ImageOps

    im = Image.open(import_file_path)
    im_invert = ImageOps.invert(im)
    im_invert.save('lena_invert.jpg', quality=95)
    im = Image.open(import_file_path).convert('RGB')
    im_invert = ImageOps.invert(im)
    im_invert.save('tt.png')
    image2 = cv2.imread('tt.png')
    cv2.imshow("Invert", image2)

    """"-----------------------------------------------"""

    img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image', img)
    #cv2.imshow('Gray image', gray)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imshow("Nosie Removal", dst)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    print("\n*********************\nImage : " + fnm + "\n*********************")
    img = cv2.imread(import_file_path)
    if img is None:
        print('no data')

    img1 = cv2.imread(import_file_path)
    print(img.shape)
    img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
    original = img.copy()
    neworiginal = img.copy()
    cv2.imshow('original', img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', img1)
   # cv2.imshow('Gray image', gray)
    p = 0
    for i in range(img.shape[0]):

        for j in range(img.shape[1]):
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if (B > 110 and G > 110 and R > 110):
                p += 1

    totalpixels = img.shape[0] * img.shape[1]
    per_white = 100 * p / totalpixels
    if per_white > 10:
        img[i][j] = [500, 300, 200]
        cv2.imshow('color change', img)
    # Guassian blur
    blur1 = cv2.GaussianBlur(img, (3, 3), 1)
    # mean-shift algo
    newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
    cv2.imshow('means shift image', img)
    # Guassian blur
    blur = cv2.GaussianBlur(img, (11, 11), 1)
    cv2.imshow('Noise Remove', blur)
    corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
    corners = np.int0(corners)

    # we iterate through each corner,
    # making a circle at each point that we think is a corner.
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, 255, -1)

    plt.imshow(image), plt.show()





def testing():
    global testing_screen
    testing_screen = Toplevel(main_screen)
    testing_screen.title("Testing")
    # login_screen.geometry("400x300")
    testing_screen.geometry("600x450+650+150")
    testing_screen.minsize(120, 1)
    testing_screen.maxsize(1604, 881)
    testing_screen.resizable(1, 1)
    # login_screen.title("New Toplevel")

    Label(testing_screen, text='''Upload Image''', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000", bg="turquoise", width="300", height="2", font=("Calibri", 16)).pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Button(testing_screen, text='''Upload Image''', font=(
        'Verdana', 15), height="2", width="30", command=imgtest).pack()

def imgtest():
    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)
    print(import_file_path)
    filename = 'data/alien_test/Test.jpg'
    cv2.imwrite(filename, image)
    print("After saving image:")
    result()

def result():
    start = time.time()

    try:

        # Path of  training images
        train_path = r'data\train'
        if not os.path.exists(train_path):
            print("No such directory")
            raise Exception
        # Path of testing images
        dir_path = r'data\alien_test'
        if not os.path.exists(dir_path):
            print("No such directory")
            raise Exception

        # Walk though all testing images one by one
        for root, dirs, files in os.walk(dir_path):
            for name in files:

                print("")
                image_path = name
                filename = dir_path + '\\' + image_path
                print(filename)
                image_size = 128
                num_channels = 3
                images = []

                if os.path.exists(filename):

                    # Reading the image using OpenCV
                    image1 = cv2.imread(filename)

                    import_file_path = filename

                    image = cv2.imread(import_file_path)
                    fnm = os.path.basename(import_file_path)
                    filename = 'Test.jpg'
                    cv2.imwrite(filename, image)
                    # print("After saving image:")

                    print("\n*********************\nImage : " + fnm + "\n*********************")
                    img = cv2.imread(import_file_path)
                    if img is None:
                        print('no data')

                    img1 = cv2.imread(import_file_path)
                    print(img.shape)
                    img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
                    original = img.copy()
                    neworiginal = img.copy()
                    cv2.imshow('original', img1)
                    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                    cv2.imshow('Original image', img1)
                    cv2.imshow('Gray image', gray)
                    p = 0
                    for i in range(img.shape[0]):

                        for j in range(img.shape[1]):
                            B = img[i][j][0]
                            G = img[i][j][1]
                            R = img[i][j][2]
                            if (B > 110 and G > 110 and R > 110):
                                p += 1

                    totalpixels = img.shape[0] * img.shape[1]
                    per_white = 100 * p / totalpixels
                    if per_white > 10:
                        img[i][j] = [500, 300, 200]
                        cv2.imshow('color change', img)
                    # Guassian blur
                    blur1 = cv2.GaussianBlur(img, (3, 3), 1)
                    # mean-shift algo
                    newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
                    cv2.imshow('means shift image', img)
                    # Guassian blur
                    blur = cv2.GaussianBlur(img, (11, 11), 1)

                    blur = cv2.GaussianBlur(img, (11, 11), 1)
                    # Canny-edge detection
                    canny = cv2.Canny(blur, 160, 290)
                    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
                    # contour to find leafs
                    bordered = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)
                    contours, hierarchy = cv2.findContours(bordered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    maxC = 0
                    for x in range(len(contours)):
                        if len(contours[x]) > maxC:
                            maxC = len(contours[x])
                            maxid = x
                    perimeter = cv2.arcLength(contours[maxid], True)
                    # print perimeter
                    Tarea = cv2.contourArea(contours[maxid])
                    cv2.drawContours(neworiginal, contours[maxid], -1, (0, 0, 255))
                    cv2.imshow('Contour', neworiginal)
                    # cv2.imwrite('Contour complete leaf.jpg',neworiginal)
                    # Creating rectangular roi around contour
                    height, width, _ = canny.shape
                    min_x, min_y = width, height
                    max_x = max_y = 0
                    frame = canny.copy()
                    # computes the bounding box for the contour, and draws it on the frame,
                    for contour, hier in zip(contours, hierarchy):
                        (x, y, w, h) = cv2.boundingRect(contours[maxid])
                        min_x, max_x = min(x, min_x), max(x + w, max_x)
                        min_y, max_y = min(y, min_y), max(y + h, max_y)
                        if w > 80 and h > 80:
                            # cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour later on
                            roi = img[y:y + h, x:x + w]
                            originalroi = original[y:y + h, x:x + w]
                    if (max_x - min_x > 0 and max_y - min_y > 0):
                        roi = img[min_y:max_y, min_x:max_x]
                        originalroi = original[min_y:max_y, min_x:max_x]
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0),
                                      2)  # we do not draw the rectangle as it interferes with contour
                    cv2.imshow('ROI', frame)
                    cv2.imshow('rectangle ROI', roi)
                    img = roi
                    # Changing colour-space
                    # imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    imghls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
                    cv2.imshow('HLS', imghls)
                    imghls[np.where((imghls == [30, 200, 2]).all(axis=2))] = [0, 200, 0]
                    cv2.imshow('new HLS', imghls)
                    # Only hue channel
                    huehls = imghls[:, :, 0]
                    cv2.imshow('img_hue hls', huehls)
                    # ret, huehls = cv2.threshold(huehls,2,255,cv2.THRESH_BINARY)
                    huehls[np.where(huehls == [0])] = [35]
                    cv2.imshow('img_hue with my mask', huehls)
                    # Thresholding on hue image
                    ret, thresh = cv2.threshold(huehls, 28, 255, cv2.THRESH_BINARY_INV)
                    cv2.imshow('thresh', thresh)






                    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                    image = cv2.resize(image1, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
                    images.append(image)
                    images = np.array(images, dtype=np.uint8)
                    images = images.astype('float32')
                    images = np.multiply(images, 1.0 / 255.0)

                    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                    x_batch = images.reshape(1, image_size, image_size, num_channels)

                    # Let us restore the saved model
                    sess = tf.compat.v1.Session()
                    # Step-1: Recreate the network graph. At this step only graph is created.
                    saver = tf.compat.v1.train.import_meta_graph('models/trained_model.meta')
                    # Step-2: Now let's load the weights saved using the restore method.
                    saver.restore(sess, tf.train.latest_checkpoint('./models/'))

                    # Accessing the default graph which we have restored
                    graph = tf.compat.v1.get_default_graph()

                    # Now, let's get hold of the op that we can be processed to get the output.
                    # In the original network y_pred is the tensor that is the prediction of the network
                    y_pred = graph.get_tensor_by_name("y_pred:0")

                    ## Let's feed the images to the input placeholders
                    x = graph.get_tensor_by_name("x:0")
                    y_true = graph.get_tensor_by_name("y_true:0")
                    y_test_images = np.zeros((1, len(os.listdir(train_path))))

                    # Creating the feed_dict that is required to be fed to calculate y_pred
                    feed_dict_testing = {x: x_batch, y_true: y_test_images}
                    result = sess.run(y_pred, feed_dict=feed_dict_testing)
                    # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                    print(result)

                    # Convert np.array to list
                    a = result[0].tolist()
                    r = 0

                    # Finding the maximum of all outputs
                    max1 = max(a)
                    index1 = a.index(max1)
                    predicted_class = None

                    # Walk through directory to find the label of the predicted output
                    count = 0
                    for root, dirs, files in os.walk(train_path):
                        for name in dirs:
                            if count == index1:
                                predicted_class = name
                            count += 1

                    # If the maximum confidence output is largest of all by a big margin then
                    # print the class or else print a warning
                    for i in a:
                        if i != max1:
                            if max1 - i < i:
                                r = 1
                    if r == 0:
                        print(predicted_class)

                        if(predicted_class=="Alpinia Galanga (Rasna)"):
                            messagebox.showinfo("Predict", predicted_class)

                            messagebox.showinfo("Uses", 'to help with inflammation, bronchitis, asthma, cough, indigestion, piles, joint pains,obesity, diabetes')

                        elif(predicted_class=="Amaranthus Viridis (Arive-Dantu)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'It is packed with carbohydrates, proteins, minerals and vitamins, and regular consumption helps in easing digestion, excessive menstruation and weight management')


                        elif (predicted_class == "Artocarpus Heterophyllus (Jackfruit)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", '‘treatment of asthma, prevent ringworm infection, and heal cracking of the feet')

                        elif (predicted_class == "Azadirachta Indica (Neem)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'Treats Acne. Neem has an anti-inflammatory property which helps reduces acne, Nourishes Skin, Treats Fungal Infections, Useful in Detoxification, Increases Immunity, Treats Wound')
                        elif (predicted_class == "Basella Alba (Basale)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'asthma, gout, pain, digestive disorders, cough, fever')

                        elif (predicted_class == "Brassica Juncea (Indian Mustard)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'Headache,toothpaste,ulcer,recurrent cold,rheumatold arthritis,joint pain')

                        elif (predicted_class == "Carissa Carandas (Karanda)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'asthma,skin disease,cough, cold and tuberculosis')

                        elif (predicted_class == "Citrus Limon (Lemon)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'rich in vitaminC,boosting immune system,strengthen blood vessels')

                        elif (predicted_class == "Ficus Auriculata (Roxburgh fig)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'leaves are crushed and applied on wounds, cure dysentery, diabetes,high cholesterol')


                        elif (predicted_class == "Ficus Religiosa (Peepal Tree)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'used for fifty types of disorders including asthma,diabetes,diarrhea,gastric problem,inflammatory disorder,,infectious')

                        elif (predicted_class == "Hibiscus Rosa-sinensis"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'treating wounds,inflamation,fever and coughs,diabetes,infection caused by bacteria and fungi, hair loss')

                        elif (predicted_class == "Jasminum (Jasmine)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", ' is an ayruvedic plant,used for treatment of wounds,skin diseases,ulcers of the oral cavity,eye disease')

                        elif (predicted_class == "Mangifera Indica (Mango)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'antiseptic, tonic, anaemia,asthma,cough,insomnia,hypertension,haemorrhage,piles')

                        elif (predicted_class == "Mentha (Mint)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'used to treat bad breath, improve brain function, cold symptoms,treat indigestion')

                        elif (predicted_class == "Moringa Oleifera (Drumstick)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'protecting and nourishing skin and hair,protection the liver,treating cancer,fighting against bacterial diseases')

                        elif (predicted_class == "Muntingia Calabura (Jamaica Cherry-Gasagase)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'antiseptic and to treat abdominal cramps and relieve headache and colds')

                        elif (predicted_class == "Murraya Koenigii (Curry)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'treating piles,inflammation,itching, dysentry,leaf infusing is drunk to stop vomiting and treat fever')

                        elif (predicted_class == "Nerium Oleander (Oleander)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'heart condition,asthma,cancer,malaria,effective for menstrual pain')

                        elif (predicted_class == "Nyctanthes Arbor-tristis (Parijata)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'anti inflammatory and antipyretic(fever-reducing) properties which help in managing pain and cough')

                        elif (predicted_class == "Ocimum Tenuiflorum (Tulsi)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'treat respiratory problems, cure fever,common cold and sore throat,headache and kidney stones')

                        elif (predicted_class == "Piper Betle (Betel)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'improves oral health, anti microbial agent, anti cancer agent, lowering high blood glucose level')

                        elif (predicted_class == "Plectranthus Amboinicus (Mexican Mint)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'defend against colds, ease the pain of arthritis, relieve stress and anxiety,treat certain kinds of cancer and optimize digestion')

                        elif (predicted_class == "Pongamia Pinnata (Indian Beech)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'treatment of tumors, piles, skin disease and ulcer')
                        elif (predicted_class == "Psidium Guajava (Guava)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'pain relief, cough, oral ulcer, effective in dysentery,good vision,manage blood sugar ')

                        elif (predicted_class == "Punica Granatum (Pomegranate)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'treating number of disorder such as insomia,abdominal pain,skin ageing and imflammation of the skin ')

                        elif (predicted_class == "Santalum Album (Sandalwood)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'treating common cold ,cough,fever and sore mouth and throat, treat urinary infection,liver disease, blood vessel')

                        elif (predicted_class == "Syzygium Cumini (Jamun)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'asthma,treatment of sore throat,asthma,thirst, It also a good blood purifier')

                        elif (predicted_class == "Syzygium Jambos (Rose Apple)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'used for digestive and tooth ailments,improve brain health, effective against small pox')

                        elif (predicted_class == "Tabernaemontana Divaricata (Crape Jasmine)"):
                            messagebox.showinfo("Result", predicted_class)
                            messagebox.showinfo("Uses", 'liver disease(hepatitis), pain due to severe diarrhea(dysentery), prevent stroke, to cause relaxationn')




                    else:
                        print("Could not classify with definite confidence")
                        print("Maybe:", predicted_class)
                        #messagebox.showinfo("Result", "Maybe:"+ predicted_class)








                else:
                    print("File does not exist")





            end = time.time()
            dur = end - start
            print("")
            if dur < 60:
                print("Execution Time:", dur, "seconds")
            elif dur > 60 and dur < 3600:
                dur = dur / 60
                print("Execution Time:", dur, "minutes")
            else:
                dur = dur / (60 * 60)
                print("Execution Time:", dur, "hours")

            # plt = plt1

            x = np.array(["Execution Time"])
            y = np.array([dur])

            # plt.set(title="Execution Time")
            plt.bar(x, y)
            plt.show()

            plt.imshow(img1)
            plt.show()
            messagebox.showinfo("Result", predicted_class)




    except Exception as e:
        print("Exception:", e)

    # Calculate execution time
'''   end = time.time()
    dur = end - start
    print("")
    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")

    # plt = plt1

    x = np.array(["Execution Time"])
    y = np.array([dur])

    # plt.set(title="Execution Time")
    plt.bar(x, y)
    plt.show()'''






def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.title("Herbal Leaf Images Classification")

    Label(text="Herbal Leaf Images Classification", bg="turquoise", width="300", height="5", font=("Calibri", 16)).pack()

    Button(text="Training", font=(
        'Verdana', 15), height="2", width="30", command=training, highlightcolor="black").pack(side=TOP)
    Label(text="").pack()
    Button(text="Testing", font=(
        'Verdana', 15), height="2", width="30", command=testing).pack(side=TOP)

    Label(text="").pack()

    main_screen.mainloop()


main_account_screen()

