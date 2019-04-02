import os
import json
import time
import warnings
import numpy as np
import PIL
from PIL import ImageTk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

from imutils import imresize, to_uint8
from xray_predictor import XrayPredictor
from xray_predictor_multi import XrayPredictorMulti


class MainMenu:
    def __init__(self, root):
        self.root: MainWindow = root
        font = self.root.gui_settings['menu_font']
        self.menu = Menu(root, font=font)
        root.config(menu=self.menu)

        submenu = Menu(self.menu, tearoff=False, font=font)
        self.menu.add_cascade(label='File', menu=submenu)
        submenu.add_command(label='Open X-ray image (Ctrl+O)', command=self.root.open_xray_image)
        self.root.bind_all("<Control-o>", self.root.open_xray_image)
        submenu.add_separator()
        submenu.add_command(label='Exit', command=root.destroy)

        submenu = Menu(self.menu, tearoff=False, font=font)
        self.menu.add_cascade(label='Help', menu=submenu)
        submenu.add_command(label='About', command=self.about)
        self.root.bind_all("<F1>", self.about)

    def about(self, event=None):
        txt = ''
        txt += 'X-ray Predictor ' + self.root.app_version + '\n\n'
        txt += 'Developed at UIIP NASB\n'
        txt += 'http://uiip.bas-net.by/\n'
        txt += 'Contact e-mail: vassili.kovalev@gmail.com\n'
        messagebox.showinfo('About', txt)


class StatusBar:
    def __init__(self, root):
        self.root: MainWindow = root
        font = self.root.gui_settings['status_font']
        self.label = Label(root, text='Initialization...', bd=1, relief=SUNKEN, anchor=W, font=font)
        self.label.pack(side=BOTTOM, fill=X)

    def set_status(self, string):
        self.label.config(text=string)
        self.root.update()


class InfoBar:
    def __init__(self, root, class_names):
        self.frame = Frame(root, bd=1, relief=GROOVE)
        self.frame.pack(side=LEFT, fill=BOTH)
        self.root = root

        self.__add_label('Image info', anchor=N)
        self.img_file_label = self.__add_label('Image file: <not loaded>')
        self.img_resolution_label = self.__add_label('Image resolution: ')
        self.__add_label('')

        self.__add_label('Prediction results', anchor=N)
        self.class_labels = {}
        if 'abnormal_lungs' not in class_names:
            class_names = ['abnormal_lungs'] + class_names

        for class_name in class_names:
            self.class_labels[class_name] = self.__add_label(class_name.replace('_', ' ') + ' : ')

    def __add_label(self, text, anchor=NW):
        width = self.root.gui_settings['panel_width']
        font = self.root.gui_settings['captions_font']

        label = Label(self.frame, width=width, text=text, anchor=anchor, justify=LEFT, bd=4, font=font)
        if anchor != NW:
            label.config(font=font + ' bold')
        label.pack(side=TOP)
        return label

    def update_info(self, file_path, img, predictions):
        text = 'Image file: ' + os.path.split(file_path)[-1]
        self.img_file_label.config(text=text)

        text = 'Image resolution: %i x %i' % (img.shape[1], img.shape[0])
        self.img_resolution_label.config(text=text)

        thresholds = {'abnormal_lungs': (0.35, 0.729),
                      'pneumonia': (0.0577, 0.164),
                      'tuberculosis': (0.064, 0.386)}

        for class_name in predictions:
            label = self.class_labels[class_name]
            p = predictions[class_name]
            text = '%s : %.02f' % (class_name.replace('_', ' '), p)

            ts = thresholds[class_name]
            if p > ts[1]:
                clr = 'red'
            elif p > ts[0]:
                clr = 'dark orange'
            else:
                clr = 'black'

            label.config(text=text, fg=clr)


class MainWindow(Tk):
    def __init__(self, setup_file_path):
        Tk.__init__(self)

        with open('gui_settings.json', 'r') as f:
            self.gui_settings = json.load(f)

        with open(setup_file_path, 'r') as f:
            self.setup = json.load(f)

        self.app_version = '0.2'
        self.title('X-ray Predictor v' + self.app_version)

        img = Image("photo", file="images/icon320a.png")
        self.tk.call('wm', 'iconphoto', self._w, img)

        self.main_menu = MainMenu(self)
        self.status_bar = StatusBar(self)
        self.status_bar.set_status('Ready')
        self.info_bar = InfoBar(self, self.setup['class_names'])

        w = self.gui_settings['max_img_width']
        h = self.gui_settings['max_img_height']
        empty_preview = PIL.Image.fromarray(np.ones((h, w, 3), dtype=np.uint8) * 127)
        empty_preview = ImageTk.PhotoImage(empty_preview)
        preview_label = Label(self, width=w, height=h, image=empty_preview, bd=2, anchor=NW, relief=SUNKEN)
        preview_label.pack(side=TOP, expand=True, fill=BOTH)
        self.preview_label = preview_label
        self.preview = empty_preview

        warnings.filterwarnings('ignore')
        # self.xray_predictor = XrayPredictor(setup_file_path)
        self.xray_predictor = XrayPredictorMulti(setup_file_path)

        self.mainloop()

    def open_xray_image(self, event=None):
        formats = '*.jpg *.png *.bmp *.jpeg *.dcm *.dicom *.JPG *.JPEG *.PNG *.BMP *.DCM *.DICOM'
        file_path = askopenfilename(initialdir='/home/skliff13/Downloads/',
                                    filetypes=(('Image or DICOM files', formats), ('All Files', '*')),
                                    title='Choose a file.')

        if file_path is None:
            return

        self.status_bar.set_status('Processing image ' + file_path)
        start = time.time()
        predictions, rgb, img_normalized = self.xray_predictor.load_and_predict_image(file_path)
        elapsed = time.time() - start
        print('Time elapsed: %.02f sec' % elapsed)
        self.status_bar.set_status('Processing finished (%.02f sec elapsed)' % elapsed)

        self.info_bar.update_info(file_path, img_normalized, predictions)
        self.update_preview(img_normalized, rgb)

    def update_preview(self, img_normalized, rgb):
        combined = rgb * 0
        for c in range(3):
            combined[:, :, c] = img_normalized
        combined = np.concatenate((combined, rgb), axis=1)

        w = self.gui_settings['max_img_width']
        h = self.gui_settings['max_img_height']
        rs = min(h / combined.shape[0], w / combined.shape[1])
        new_shape = (int(combined.shape[0] * rs), int(combined.shape[1] * rs))
        combined = imresize(combined, new_shape)
        combined = to_uint8(combined)

        preview = PIL.Image.fromarray(combined)
        preview = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=preview)
        self.preview = preview


if __name__ == '__main__':
    MainWindow('setup_vgg16m_1.json')
