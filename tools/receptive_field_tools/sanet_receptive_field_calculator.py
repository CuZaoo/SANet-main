"""
striden = strides= stride1*stride2*...*striden-1
rn = rn-1 + (fn - 1)*strides

# 卷积核大小/stride/padding
# 卷积核大小/stride/padding/dilation
"""
# (k,s,p,d) = (3, 2, 1, 1)
net_struct = {
    'regseg': {'net': [
        # 打开下面的注释得到的感受野是3807，这和regseg官方的感受野一致，但是这不符合regseg的网络结构
        # [3, 2, 1, 1],
        # [3, 2, 1, 1],
        # [3, 2, 1, 1],
        # [3, 2, 1, 1],
        #
        # [3, 1, 1, 1],

        # [3, 1, 1, 1],
        # [3, 1, 1, 2],
        # [3, 1, 1, 4],
        # [3, 1, 1, 4],
        # [3, 1, 1, 4],
        # [3, 1, 1, 4],
        #
        # [3, 1, 1, 14],
        # [3, 1, 1, 14],
        # [3, 1, 1, 14],
        # [3, 1, 1, 14],
        # [3, 1, 1, 14],
        # [3, 1, 1, 14],
        # [3, 1, 1, 14]

        # 按照regseg网络结构的感受野
        [3, 2, 1, 1],

        [3, 2, 1, 1],
        [3, 1, 1, 1],

        [3, 2, 1, 1],
        [3, 1, 1, 1],
        [3, 1, 1, 1],
        [3, 1, 1, 1],
        [3, 1, 1, 1],
        [3, 1, 1, 1],

        [3, 2, 1, 1],
        # 最后一层
        [3, 1, 1, 1],
        [3, 1, 2, 2],
    ]+ [[3, 1, 4, 4].copy() for _ in range(4)] + [[3, 1, 14, 14].copy() for _ in range(7)]},
    # 3x3卷积影响感受野，1x1不影响
    "SANet_主干": {'net': [
        [3, 2, 1, 1],
        [3, 2, 1, 1],

        [3, 1, 1, 1],
        [3, 1, 1, 1],

        [3, 2, 1, 1],
        [3, 1, 1, 1],

        [3, 1, 1, 1],
        [3, 1, 1, 1],

        [3, 2, 1, 1],
        [3, 1, 1, 1],

        [3, 2, 1, 1],
        [3, 1, 1, 1],

        [3, 2, 1, 1],
        [3, 1, 1, 1],

        # pool
        [5, 2, 2, 1],
        [9, 4, 4, 1],
        [17, 8, 8, 1],
    ]},"SANet_分支": {'net': [
        [3, 2, 1, 1],
        [3, 2, 1, 1],

        [3, 1, 1, 1],
        [3, 1, 1, 1],

        [3, 2, 1, 1],
        [3, 1, 1, 1],

        [3, 1, 1, 1],
        [3, 1, 1, 1],



        [3, 1, 1, 1],
        [3, 1, 2, 2],
        [3, 1, 5, 5],

        [3, 1, 7, 7],
        [3, 1, 13, 13],

        # [3, 2, 1,/ 1],

    ]}
}


def calc_respective_fields(net):
    layers = net['net']
    layers_num = len(layers)
    result = []
    rf = 1
    strides = 1
    for i in range(layers_num):
        length = len(layers[i])
        if 3 == length:
            f, s, p = layers[i]
            d = 1
        elif 4 == length:
            f, s, p, d = layers[i]
        else:
            print("len(layers[i]) should = 3 or 4!\n ")
            exit(-1)

        # 扩大卷积
        f = (f - 1) * d + 1
        rf = rf + (f - 1) * strides
        strides *= s
        result.append([rf, strides])
    return result


if __name__ == '__main__':
    # 输入参数'regseg','SANet_主干','SANet_分支'或者自行构建你自己的网络,注意:strides会有错误,比如strides太大
    net = net_struct['regseg']
    result = calc_respective_fields(net)
    for i in range(len(result)):
        print(' layer output respective field %s strides %s' % (result[i][0], result[i][1]))
