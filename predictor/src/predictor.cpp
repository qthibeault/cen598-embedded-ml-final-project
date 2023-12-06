#include <array>
#include <iostream>

#include "nanobind/nanobind.h"
#include "nanobind/stl/array.h"
#include "tensorflow/lite/c/c_api.h"

extern unsigned char load_model_tflite[];
extern unsigned int load_model_tflite_len;

extern unsigned char power_model_tflite[];
extern unsigned int power_model_tflite_len;

static TfLiteModel* load_model = TfLiteModelCreate(load_model_tflite, load_model_tflite_len);
static TfLiteInterpreterOptions* load_interp_opts = TfLiteInterpreterOptionsCreate();
static TfLiteInterpreter* load_interp = TfLiteInterpreterCreate(load_model, load_interp_opts);

static TfLiteModel* power_model = TfLiteModelCreate(power_model_tflite, power_model_tflite_len);
static TfLiteInterpreterOptions* power_interp_opts = TfLiteInterpreterOptionsCreate();
static TfLiteInterpreter* power_interp = TfLiteInterpreterCreate(power_model, power_interp_opts);


int predict_load(const std::array<float, 4>& samples)
{
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(load_interp, 0);
    TfLiteTensorCopyFromBuffer(input_tensor, samples.data(), samples.size() * sizeof(float));
    TfLiteInterpreterInvoke(load_interp);

    std::array<float, 2> output;
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(load_interp, 0);
    TfLiteTensorCopyToBuffer(output_tensor, output.data(), output.size() * sizeof(float));

    if (output[1] > output[0]) {
        return 1;
    }

    return -1;
}

bool predict_active(const std::array<float, 4>& samples)
{
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(power_interp, 0);
    TfLiteTensorCopyFromBuffer(input_tensor, samples.data(), samples.size() * sizeof(float));
    TfLiteInterpreterInvoke(power_interp);

    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(power_interp, 0);

    std::array<float, 2> output;
    TfLiteTensorCopyToBuffer(output_tensor, output.data(), output.size() * sizeof(float));

    if (output[1] > output[0]) {
        return true;
    }

    return false;
}

NB_MODULE(predictor, m) {
    TfLiteInterpreterAllocateTensors(load_interp);
    TfLiteInterpreterAllocateTensors(power_interp);

    m.def("predict_load", &predict_load);
    m.def("predict_active", &predict_active);
}
