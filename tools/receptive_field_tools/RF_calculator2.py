'''
    File      [ receptive_field_calculator.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
'''


def conv1d_r(r_out, kernel_size, stride, dilation=1):
    ''' Computes receptive field size berfore a conv1d layer. '''
    if dilation == 1:
        return r_out * stride + max(kernel_size - stride, 0)
    else:
        kernel_size += dilation
        return r_out * stride + max(kernel_size - stride, 0)

def conv2d_r(r_out, kernel_size, stride, dilation=1):
    ''' Computes receptive field size berfore a conv2d layer. '''

    assert isinstance(r_out, tuple)
    assert isinstance(kernel_size, (int, tuple))
    assert isinstance(stride, (int, tuple))
    assert isinstance(dilation, (int, tuple))

    kernel_0 = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    kernel_1 = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
    stride_0 = stride if isinstance(stride, int) else stride[0]
    stride_1 = stride if isinstance(stride, int) else stride[1]
    dilation_0 = dilation if isinstance(dilation, int) else dilation[0]
    dilation_1 = dilation if isinstance(dilation, int) else dilation[1]

    return (conv1d_r(r_out[0], kernel_0, stride_0, dilation_0),
            conv1d_r(r_out[1], kernel_1, stride_1, dilation_1))


def receptive_field_calculator(layers):
    '''
        The receptive field calculator.
        Input:
            layers [list]: a list of layers, each consisting of
                its (1) layer type, (2) kernel size, (3) stride,
                and (4) dilation.
    '''

    # Compute receptive field
    if layers[0][0] == 'conv1d':
        r_field = [1]
    else:
        r_field = [(1, 1)]

    for i, (layer_type, kernel_size, stride, dilation) in \
            reversed(list(enumerate(layers))):
        if layer_type == 'conv1d':
            r_field.append(
                conv1d_r(r_field[-1], kernel_size, stride, dilation))
        elif layer_type == 'conv2d':
            r_field.append(
                conv2d_r(r_field[-1], kernel_size, stride, dilation))
        else:
            raise ValueError(f'Unknown layer type {layer_type}')

    # Print results
    format_str = ' {:<6} {:<10} {:<8} {:<8} {:<10} {:<15}'
    print('-' * 61)
    print(format_str
          .format('layer', 'type', 'kernel', 'stride', 'dilation', 'r field'))
    print('-' * 61)
    for i, (layer_type, kernel_size, stride, dilation) in enumerate(layers):
        print(format_str
              .format(i + 1, layer_type,
                      str(kernel_size), str(stride), str(dilation),
                      str(r_field[-(i + 1)])))
    print('-' * 61)


if __name__ == '__main__':
    # Debug code
    # layer_type, kernel_size, stride, dilation
    test_layers = [
        # ('conv1d', 3, 1, 2),
        # ('conv1d', 3, 1, 2),


        ('conv1d', 3, 2, 1),

        ('conv1d', 3, 2, 1),

        ('conv1d', 3, 2, 1),
        ('conv1d', 3, 1, 1),
        ('conv1d', 3, 1, 1),

        ('conv1d', 3, 2, 1),
        ('conv1d', 3, 1, 1),

        ('conv1d', 3, 1, 2),

        ('conv1d', 3, 1, 4),
        ('conv1d', 3, 1, 4),
        ('conv1d', 3, 1, 4),
        ('conv1d', 3, 1, 4),

        ('conv1d', 3, 1, 14),
        ('conv1d', 3, 1, 14),
        ('conv1d', 3, 1, 14),
        ('conv1d', 3, 1, 14),
        ('conv1d', 3, 1, 14),
        ('conv1d', 3, 1, 14),

        ('conv1d', 3, 1, 14),

    ]

    receptive_field_calculator(test_layers)

    # 321
    # 321
    # 311

