import math

def hsi2rgb(h, s, i):
    h = float(h)
    s = float(s)
    i = float(i)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = i * (1 - s)
    q = i * (1 - f * s)
    t = i * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = i, t, p
    elif hi == 1: r, g, b = q, i, p
    elif hi == 2: r, g, b = p, i, t
    elif hi == 3: r, g, b = p, q, i
    elif hi == 4: r, g, b = t, p, i
    elif hi == 5: r, g, b = i, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
    
def rgb2hsi(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    i = mx
    return h, s, i

def rgb2hex(r,g,b):
    return '#{:02x}{:02x}{:02x}'.format( r, g , b )