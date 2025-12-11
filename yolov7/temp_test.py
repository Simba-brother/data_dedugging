
def test_1():
    data = [3,4,2,1]
    b = data[[0,8]]
    print(b)


def test_2():
    a = {1:"mml"}
    b = a.get(2)
    print()

def xcycwh_to_x1y1x2y2(bbox,W,H):
    xc = bbox[0]
    yc = bbox[1]
    w = bbox[2]
    h = bbox[3]

    # 1. 归一化 -> 像素
    x_c = xc * W
    y_c = yc * H
    bw  = w  * W
    bh  = h  * H

    # 2. 中心 -> 左上 / 右下
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2

    # 3. 转 int + 裁剪
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))

    return [x1,y1,x2,y2]

def test_3():
    data = [1]
    b = data[:-1]
    print(b)

def test_4():
    a = [1,"b.j"]
    print(a)

def test_5():
    a = ["a","b","c"]
    print(a[-2:])
def test_6():
    data = [1]*10
    print(data)
if __name__ == "__main__":
    # test_2()
    test_6()
    # bbox = [0.499,0.4866666666666667,0.106,0.14666666666666667]
    # W = 500
    # H = 375
    # b = xcycwh_to_x1y1x2y2(bbox,W,H)
    # print()