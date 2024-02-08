#    |\---/|
#    | ,_, |
#     \_`_/-..----.
#  ___/ `   ' ,""+ \  zzzzzzzz
# (__...'   __\    |`.___.';
#   (_,...'(_,.`__)/'.....+
# Fernando Araujo Alves Filho

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def reduce_color_quantization(self):
        N = 64
        return (self.image // N) * N

    def reduce_color_kmeans(self):
        K = 8
        n = self.image.shape[0] * self.image.shape[1]
        data = self.image.reshape(n, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

        data = centers[labels.flatten()].reshape(self.image.shape)
        reduced = data.astype(np.uint8)
        return reduced

    def reduce_color_stylization(self):
        return cv2.stylization(self.image)

    def reduce_color_edge_preserving(self):
        return cv2.edgePreservingFilter(self.image)

    def get_palette(self, src):
        palette = {}
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                color = tuple(src[r, c])
                if color not in palette:
                    palette[color] = 1
                else:
                    palette[color] += 1
        return palette

    def print_palette(self, palette, area):
        for color, count in palette.items():
            percentage = 100.0 * count / area
            print(f"Color: {color} - Area: {percentage:.2f}%")

    def process_image(self, reduction_method):
        reduced = reduction_method()
        palette = self.get_palette(reduced)
        area = self.image.shape[0] * self.image.shape[1]
        self.print_palette(palette, area)
    def normalize_colors(self, colors):
        return [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

    def save_palette_pie_chart(self, palette, path_chart):
        counts = list(palette.values())
        colors = self.normalize_colors(list(palette.keys()))
        
        plt.pie(counts, colors=colors, startangle=140)
        plt.title("Paleta de Cores da Imagem")
        plt.axis('equal')  
        plt.savefig(path_chart)
        plt.show()

    def process_image_with_palette_chart(self, reduction_method, path_chart= 'palette_pie_chart.png'):
        reduced = reduction_method()
        palette = self.get_palette(reduced)
        area = self.image.shape[0] * self.image.shape[1]
        self.print_palette(palette, area)

        # Save the palette as a pie chart
        self.save_palette_pie_chart(palette, path_chart)
     
     
# usage_example    
if __name__ == "__main__":
    image_processor = ImageProcessor("path_to_image")

    # Process image using different reduction methods
    image_processor.process_image(image_processor.reduce_color_quantization)
    image_processor.process_image(image_processor.reduce_color_kmeans)
    image_processor.process_image(image_processor.reduce_color_stylization)
    image_processor.process_image(image_processor.reduce_color_edge_preserving)
