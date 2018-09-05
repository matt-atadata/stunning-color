import math
import cv2
import numpy as np
import skimage.color as skc
import skimage.filters as skf
import matplotlib.pyplot as plt
import skimage.morphology as skm

from PIL import Image
from skimage.measure import label
from skimage.io import imread, imsave
from skimage.util import img_as_float



class cached_property(object):
    """Decorator that creates converts a method with a single
    self argument into a property cached on the instance.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type):
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res


class ColorMe(object):
    """ColorMe main class. Sections of code from ColorThief-py codebase."""
    def __init__(self, file):
        """Create one color thief for one image.

        :param file: A filename (string) or a file object. The file object
                     must implement `read()`, `seek()`, and `tell()` methods,
                     and be opened in binary mode.
        """
        # self.resized, self.back_mask, self.img_back_mask = self.getMasks(file)
        #  * ~np.reshape(self.img_back_mask, self.img_back_mask.shape + (1,))   # Adding third dimension to match rgb colorspace: numpy.reshape(array, array.shape + (1,))
        self.image =  Image.open(file) 

    # Pure Python, usable speed but over 10x greater runtime than Cython version
    def fill(self, data, start_coords, fill_value):
        """
        Flood fill algorithm
        
        Parameters
        ----------
        data : (M, N) ndarray of uint8 type
            Image with flood to be filled. Modified inplace.
        start_coords : tuple
            Length-2 tuple of ints defining (row, col) start coordinates.
        fill_value : int
            Value the flooded area will take after the fill.
            
        Returns
        -------
        None, ``data`` is modified inplace.
        """

        xsize, ysize = data.shape;
        orig_value = data[start_coords[0], start_coords[1]]
        
        stack = set(((start_coords[0], start_coords[1]),))
        if fill_value == orig_value:
            raise ValueError("Filling region with same value "
                             "already present is unsupported. "
                             "Did you already fill this region?")

        while stack:
            x, y = stack.pop()

            if data[x, y] == orig_value:
                data[x, y] = fill_value
                if x > 0:
                    stack.add((x - 1, y))
                if x < (xsize - 1):
                    stack.add((x + 1, y))
                if y > 0:
                    stack.add((x, y - 1))
                if y < (ysize - 1):
                    stack.add((x, y + 1))

    def _scharr(self, img):
        # Invert the image to ease edge detection.
        img = 1. - img
        grey = skc.rgb2grey(img)
        return skf.scharr(grey)

    def _floodfill(self, img):
        """
            Flood-fill background removal image by foreground selection
        """
        back = self._scharr(img)
        # Binary thresholding.
        back = back > 2.5

        # Thin all edges to be 1-pixel wide.
        back = skm.skeletonize(back)

        # Edges are not detected on the borders, make artificial ones.
        back[0, :] = back[-1, :] = True
        back[:, 0] = back[:, -1] = True

        # Label adjacent pixels of the same color.
        labels = label(back, background=-1, connectivity=1)

        # Count as background all pixels labeled like one of the corners.
        corners = [(1, 1), (-2, 1), (1, -2), (-2, -2)]
        for l in (labels[i, j] for i, j in corners):
            back[labels == l] = True

        # Remove remaining inner edges.
        return skm.opening(back)

    def _global(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.bool)
        max_distance = 5. # 5

        img = skc.rgb2lab(img)

        # Compute euclidean distance of each corner against all other pixels.
        corners = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
        for color in (img[i, j] for i, j in corners):
            norm = np.sqrt(np.sum(np.square(img - color), 2))
            # Add to the mask pixels close to one of the corners.
            mask |= norm < max_distance

        return mask

    def get_foreground(self, img):
        f = self._floodfill(img)
        g = self._global(img)
        m = f | g

        if np.count_nonzero(m) < 0.90 * m.size:
            return m

        ng = np.count_nonzero(g)
        nf = np.count_nonzero(f)

        if ng < 0.90 * g.size and nf < 0.90 * f.size:
            return g if ng > nf else f

        if ng < 0.90 * g.size:
            return g

        if nf < 0.90 * f.size:
            return f

        return np.zeros_like(m)

        
    def remove_background(self, img):
        print("Removing  background")
        #== Parameters =======================================================================
        BLUR = 21
        CANNY_THRESH_1 = 5
        CANNY_THRESH_2 = 10
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        MASK_COLOR = (255,255,255) # In BGR format


        #== Edge Background Removal ==========================================================

        #-- Read image -----------------------------------------------------------------------
        arr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bg = cv2.cvtColor(arr_img, cv2.COLOR_RGB2RGBA)
        gray = cv2.cvtColor(arr_img,cv2.COLOR_BGR2GRAY)

        #-- Edge detection -------------------------------------------------------------------
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        #-- Find contours in edges, sort by area ---------------------------------------------
        contour_info = []
        _, contours, _= cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]

        #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255,255,255))

        #-- Smooth mask, then blur it --------------------------------------------------------
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)


        #== Flood-fill background removal =====================================================
        im_floodfill = arr_img.copy()

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Create mask of image forground for keeping
        mask_flood = self.get_foreground(im_floodfill_inv)

        # Combine Edge and Flood-fill masks
        combined = cv2.subtract(mask.astype("uint8"),mask_flood.astype("uint8"))

        # Apply combined mask to alpha channel of image
        bg[:, :, 3] = combined
    
        # Apply mask for visual checkpoint to see success of background removal
        bg_visual = cv2.bitwise_and(arr_img, arr_img, mask=mask.astype("uint8"))
        fg_visual = cv2.bitwise_or(arr_img, arr_img, mask=mask_flood.astype("uint8"))
        combined_visual = cv2.subtract(bg_visual,fg_visual)

        # Save test images
        cv2.imwrite('./bg.jpg', bg_visual)
        cv2.imwrite('./fg.jpg', fg_visual)
        cv2.imwrite('./combined_masked.jpg', combined_visual)

        return bg
    
    def remove_skin(self, img):
        # Convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the specified upper and lower boundaries converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        print('removing skin')
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
        
        # Apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # Blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(img, img, mask=~skinMask)
        cv2.imwrite('/Users/mstrautmann/Documents/colortest/map_my_color/skin_mask.jpg', skinMask)
        cv2.imwrite('/Users/mstrautmann/Documents/colortest/map_my_color/skin.jpg', skin)
        return skin

    def array2PIL(self, arr, size):
        """
            Helper function to take raw img array and convert to PIL library byteIO format
        """
        mode = 'RGBA'
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
        if len(arr[0]) == 3:
            arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
        return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)
        
    def get_palette(self, color_count=10, quality=1):
        """Build a color palette.  We are using the median cut algorithm to
        cluster similar colors.

        :param color_count: the size of the palette, max number of colors
        :param quality: quality settings, 1 is the highest quality, the bigger
                        the number, the faster the palette generation, but the
                        greater the likelihood that colors will be missed.
        :return list: a list of tuple in the form (r, g, b)
        """
        arr_img = self.remove_background(self.image.convert('RGBA'))
        # arr_img = cv2.blur(arr_img,(5,5))
        #arr_img = self.remove_skin(arr_img)
        image_rgb = cv2.cvtColor(np.array(arr_img), cv2.COLOR_BGR2RGB)
        cv2.imwrite('./image_final.jpg', (cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR))) 
        image_rgba = self.array2PIL(image_rgb, self.image.size).convert('RGBA')
        width, height = self.image.size
        pixels = image_rgba.getdata()
        pixel_count = width * height
        
        valid_pixels = []
        for i in range(0, pixel_count, quality):
            r, g, b, a = pixels[i]
            # If pixel is mostly opaque and not white
            if a >= 125:
                if not (r > 250 and g > 250 and b > 250):
                    valid_pixels.append((r, g, b))

        # Send array to quantize function which clusters values
        # using median cut algorithm
        cmap = MMCQ.quantize(valid_pixels, color_count)
        return cmap.palette
        
    def get_color(self, quality=1):
        """Get the dominant color.

        :param quality: quality settings, 1 is the highest quality, the bigger
                        the number, the faster a color will be returned but
                        the greater the likelihood that it will not be the
                        visually most dominant color
        :return tuple: (r, g, b)
        """
        palette = self.get_palette(5, quality)
        return palette

class MMCQ(object):
    """Basic Python port of the MMCQ (modified median cut quantization)
    algorithm from the Leptonica library (http://www.leptonica.com/).
    """

    SIGBITS = 5
    RSHIFT = 8 - SIGBITS
    MAX_ITERATION = 1000
    FRACT_BY_POPULATIONS = 0.75

    @staticmethod
    def get_color_index(r, g, b):
        return (r << (2 * MMCQ.SIGBITS)) + (g << MMCQ.SIGBITS) + b

    @staticmethod
    def get_histo(pixels):
        """histo (1-d array, giving the number of pixels in each quantized
        region of color space)
        """
        histo = dict()
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            index = MMCQ.get_color_index(rval, gval, bval)
            histo[index] = histo.setdefault(index, 0) + 1
        return histo

    @staticmethod
    def vbox_from_pixels(pixels, histo):
        rmin = 1000000
        rmax = 0
        gmin = 1000000
        gmax = 0
        bmin = 1000000
        bmax = 0
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            rmin = min(rval, rmin)
            rmax = max(rval, rmax)
            gmin = min(gval, gmin)
            gmax = max(gval, gmax)
            bmin = min(bval, bmin)
            bmax = max(bval, bmax)
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax, histo)

    @staticmethod
    def median_cut_apply(histo, vbox):
        if not vbox.count:
            return (None, None)

        rw = vbox.r2 - vbox.r1 + 1
        gw = vbox.g2 - vbox.g1 + 1
        bw = vbox.b2 - vbox.b1 + 1
        maxw = max([rw, gw, bw])
        # only one pixel, no split
        if vbox.count == 1:
            return (vbox.copy, None)
        # Find the partial sum arrays along the selected axis.
        total = 0
        sum_ = 0
        partialsum = {}
        lookaheadsum = {}
        do_cut_color = None
        if maxw == rw:
            do_cut_color = 'r'
            for i in range(vbox.r1, vbox.r2+1):
                sum_ = 0
                for j in range(vbox.g1, vbox.g2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(i, j, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        elif maxw == gw:
            do_cut_color = 'g'
            for i in range(vbox.g1, vbox.g2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(j, i, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        else:  # maxw == bw
            do_cut_color = 'b'
            for i in range(vbox.b1, vbox.b2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.g1, vbox.g2+1):
                        index = MMCQ.get_color_index(j, k, i)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        for i, d in partialsum.items():
            lookaheadsum[i] = total - d

        # determine the cut planes
        dim1 = do_cut_color + '1'
        dim2 = do_cut_color + '2'
        dim1_val = getattr(vbox, dim1)
        dim2_val = getattr(vbox, dim2)
        for i in range(dim1_val, dim2_val+1):
            if partialsum[i] > (total / 2):
                vbox1 = vbox.copy
                vbox2 = vbox.copy
                left = i - dim1_val
                right = dim2_val - i
                if left <= right:
                    d2 = min([dim2_val - 1, int(i + right / 2)])
                else:
                    d2 = max([dim1_val, int(i - 1 - left / 2)])
                # avoid 0-count boxes
                while not partialsum.get(d2, False):
                    d2 += 1
                count2 = lookaheadsum.get(d2)
                while not count2 and partialsum.get(d2-1, False):
                    d2 -= 1
                    count2 = lookaheadsum.get(d2)
                # set dimensions
                setattr(vbox1, dim2, d2)
                setattr(vbox2, dim1, getattr(vbox1, dim2) + 1)
                return (vbox1, vbox2)
        return (None, None)

    @staticmethod
    def quantize(pixels, max_color):
        """Quantize.

        :param pixels: a list of pixel in the form (r, g, b)
        :param max_color: max number of colors
        """
        if not pixels:
            raise Exception('Empty pixels when quantize.')
        if max_color < 2 or max_color > 256:
            raise Exception('Wrong number of max colors when quantize.')

        histo = MMCQ.get_histo(pixels)

        # check that we aren't below maxcolors already
        if len(histo) <= max_color:
            # generate the new colors from the histo and return
            pass

        # get the beginning vbox from the colors
        vbox = MMCQ.vbox_from_pixels(pixels, histo)
        pq = PQueue(lambda x: x.count)
        pq.push(vbox)

        # inner function to do the iteration
        def iter_(lh, target):
            n_color = 1
            n_iter = 0
            while n_iter < MMCQ.MAX_ITERATION:
                vbox = lh.pop()
                if not vbox.count:  # just put it back
                    lh.push(vbox)
                    n_iter += 1
                    continue
                # do the cut
                vbox1, vbox2 = MMCQ.median_cut_apply(histo, vbox)
                if not vbox1:
                    raise Exception("vbox1 not defined; shouldn't happen!")
                lh.push(vbox1)
                if vbox2:  # vbox2 can be null
                    lh.push(vbox2)
                    n_color += 1
                if n_color >= target:
                    return
                if n_iter > MMCQ.MAX_ITERATION:
                    return
                n_iter += 1

        # first set of colors, sorted by population
        iter_(pq, MMCQ.FRACT_BY_POPULATIONS * max_color)

        # Re-sort by the product of pixel occupancy times the size in
        # color space.
        pq2 = PQueue(lambda x: x.count * x.volume)
        while pq.size():
            pq2.push(pq.pop())

        # next set - generate the median cuts using the (npix * vol) sorting.
        iter_(pq2, max_color - pq2.size())

        # calculate the actual colors
        cmap = CMap()
        while pq2.size():
            cmap.push(pq2.pop())
        return cmap


class VBox(object):
    """3d color space box"""
    def __init__(self, r1, r2, g1, g2, b1, b2, histo):
        self.r1 = r1
        self.r2 = r2
        self.g1 = g1
        self.g2 = g2
        self.b1 = b1
        self.b2 = b2
        self.histo = histo

    @cached_property
    def volume(self):
        sub_r = self.r2 - self.r1
        sub_g = self.g2 - self.g1
        sub_b = self.b2 - self.b1
        return (sub_r + 1) * (sub_g + 1) * (sub_b + 1)

    @property
    def copy(self):
        return VBox(self.r1, self.r2, self.g1, self.g2,
                    self.b1, self.b2, self.histo)

    @cached_property
    def avg(self):
        ntot = 0
        mult = 1 << (8 - MMCQ.SIGBITS)
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    histoindex = MMCQ.get_color_index(i, j, k)
                    hval = self.histo.get(histoindex, 0)
                    ntot += hval
                    r_sum += hval * (i + 0.5) * mult
                    g_sum += hval * (j + 0.5) * mult
                    b_sum += hval * (k + 0.5) * mult

        if ntot:
            r_avg = int(r_sum / ntot)
            g_avg = int(g_sum / ntot)
            b_avg = int(b_sum / ntot)
        else:
            r_avg = int(mult * (self.r1 + self.r2 + 1) / 2)
            g_avg = int(mult * (self.g1 + self.g2 + 1) / 2)
            b_avg = int(mult * (self.b1 + self.b2 + 1) / 2)

        return r_avg, g_avg, b_avg

    def contains(self, pixel):
        rval = pixel[0] >> MMCQ.RSHIFT
        gval = pixel[1] >> MMCQ.RSHIFT
        bval = pixel[2] >> MMCQ.RSHIFT
        return all([
            rval >= self.r1,
            rval <= self.r2,
            gval >= self.g1,
            gval <= self.g2,
            bval >= self.b1,
            bval <= self.b2,
        ])

    @cached_property
    def count(self):
        npix = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    index = MMCQ.get_color_index(i, j, k)
                    npix += self.histo.get(index, 0)
        return npix


class CMap(object):
    """Color map"""
    def __init__(self):
        self.vboxes = PQueue(lambda x: x['vbox'].count * x['vbox'].volume)

    @property
    def palette(self):
        return self.vboxes.map(lambda x: x['color'])

    def push(self, vbox):
        self.vboxes.push({
            'vbox': vbox,
            'color': vbox.avg,
        })

    def size(self):
        return self.vboxes.size()

    def nearest(self, color):
        d1 = None
        p_color = None
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            d2 = math.sqrt(
                math.pow(color[0] - vbox['color'][0], 2) +
                math.pow(color[1] - vbox['color'][1], 2) +
                math.pow(color[2] - vbox['color'][2], 2)
            )
            if d1 is None or d2 < d1:
                d1 = d2
                p_color = vbox['color']
        return p_color

    def map(self, color):
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            if vbox['vbox'].contains(color):
                return vbox['color']
        return self.nearest(color)


class PQueue(object):
    """Simple priority queue."""
    def __init__(self, sort_key):
        self.sort_key = sort_key
        self.contents = []
        self._sorted = False

    def sort(self):
        self.contents.sort(key=self.sort_key)
        self._sorted = True

    def push(self, o):
        self.contents.append(o)
        self._sorted = False

    def peek(self, index=None):
        if not self._sorted:
            self.sort()
        if index is None:
            index = len(self.contents) - 1
        return self.contents[index]

    def pop(self):
        if not self._sorted:
            self.sort()
        return self.contents.pop()

    def size(self):
        return len(self.contents)

    def map(self, f):
        return list(map(f, self.contents))
