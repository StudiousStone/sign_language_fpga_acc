import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format
from shutil import copyfile
import sys

def generate_layer_definition(net,model_name,wei_frac,lay_frac_out):
    base_dir = "/home/localhost/pa1850co-s/tests-caffe/classify-image/"
    output_name = base_dir + "layer_definition.h"
    template = base_dir + "template.h"

    model = caffe_pb2.NetParameter()
    model_f = open(model_name, 'r')
    model = google.protobuf.text_format.Merge(str(model_f.read()), model)
    model_f.close()

    copyfile(template,output_name)
    output = open(output_name,"a")

    name = ""
    print_name = ""
    lay_type = ""
    stride = 0
    lay_ch_out = 0
    lay_ch_in = 0
    lay_px_in = 0
    lay_px_out = 0
    activations_bw = 0
    frac_in = 0
    conv1_in_frac = 0
    frac_out = 0
    conv10_out_frac = 0
    frac_wei = 0
    lay_count = 0

    image_px = net.blobs["data"].shape[2]
    image_ch = net.blobs["data"].shape[1]
    output_px = net.blobs["conv10_transfer"].shape[2]
    output_ch = net.blobs["conv10_transfer"].shape[1]
    max_output_size = 0
    max_input_size = 0
    bias_count = 0
    weights_count = 0 #Divided by 9
    max_ch = 0

    net_def = ""
    
    for i in range(0, len(model.layer)):
        if model.layer[i].type == 'ConvolutionRistretto':
            lay_count = lay_count + 1
            #Get name in "c compatible way" for printing
            name = model.layer[i].name
            print_name = name.replace(".","_")
            print_name = name.replace("/","_")
            print_name = print_name.upper()
            net_def = net_def + print_name + ","
        



            #stride
            stride = model.layer[i].convolution_param.stride
            if len(stride) > 0:
                stride = stride[0]
            else:
                stride = 1
                
            #channels and pixels
            lay_ch_out = net.params[name][0].shape[0]
            lay_ch_in = net.params[name][0].shape[1]
            lay_px_out = net.blobs[name].shape[2]
            lay_px_in = lay_px_out * stride
                
            #fractional parts
            frac_in = model.layer[i].quantization_param.fl_layer_in
            frac_out = model.layer[i].quantization_param.fl_layer_out
            frac_wei = model.layer[i].quantization_param.fl_params
            #   Used to send the info about fractional parts to the calling function
            wei_frac.append(frac_wei)
            lay_frac_out.append(frac_out)
            if name == "conv1":
                conv1_in_frac = frac_in
                activations_bw =  model.layer[i].quantization_param.bw_layer_in
                
            #weights size
            if model.layer[i].convolution_param.kernel_size[0] == 3:
                lay_type = "CONV_3x3"
                weights_count = weights_count + (lay_ch_in * lay_ch_out)
            else:
                lay_type = "CONV_1x1"
                weights_count = weights_count + ((lay_ch_in * lay_ch_out) / 9) + 1
            bias_count = bias_count + lay_ch_out

            #mem_i and mem_o vectors size
            if lay_ch_out * lay_px_out * lay_px_out > max_output_size:
                max_output_size = lay_ch_out * lay_px_out * lay_px_out
            if lay_ch_in * lay_px_in * lay_px_in > max_input_size:
                max_input_size = lay_ch_in * lay_px_in * lay_px_in
            if lay_ch_out > max_ch:
                max_ch = lay_ch_out
            if lay_ch_in > max_ch:
                max_ch = lay_ch_in

            #print layer definition
            output.write("#define " + print_name + "  {" + \
                str(lay_count) + "," + str(lay_type) + "," + \
                str(stride) + "," + str(lay_ch_in) + "," + \
                str(lay_ch_out) + "," + str(lay_px_in) + "," + \
                str(lay_px_out) + "," + str(frac_in) + "," + \
                str(frac_out) + "," + str(frac_wei) + "}" + "\n")
            model.layer[i].quantization_param.fl_layer_in
            
    
    
    output.write("#define NET " + "{" + net_def + "}")
    output.write("\n\n")

    #Print some extra useful info 
    output.write("#define IM_PX_SZ " + str(image_px) + "\n")
    output.write("#define IM_CH_IN " + str(image_ch) + "\n")
    output.write("#define OUT_PX " + str(output_px) + "\n")
    output.write("#define OUT_CH " + str(output_ch) + "\n\n")
    output.write("#define MAX_OUTPUT_SIZE " + str(max_output_size) + "\n")
    output.write("#define MAX_INPUT_SIZE " + str(max_input_size) + "\n")
    output.write("#define TOTAL_BIAS " + str(bias_count) + "\n")
    output.write("#define TOTAL_WEIGHTS " + str(weights_count) + "\n")
    output.write("#define MAX_CH_OUT " + str(max_ch) + "\n\n")
    output.write("#define TOTAL_OPS " + str(lay_count) + "\n")

    output.close()
    
    return conv1_in_frac, activations_bw

if __name__ == '__main__':
    generate_layer_definition(sys.argv)
