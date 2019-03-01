class Cropping:
    def __init__(self, x_low, x_high, y_low, y_high):
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high

    def crop_image(self, img):
        return img[self.y_low:self.y_high, self.x_low:self.x_high]

    def unpack_values(self):
        return self.x_low, self.x_high, self.y_low, self.y_high
