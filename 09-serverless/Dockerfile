FROM public.ecr.aws/lambda/python:3.8

RUN pip install --no-cache-dir keras-image-helper \
    && pip install --no-cache-dir https://github.com/alexeygrigorev/tflite-aws-lambda/raw/refs/heads/main/tflite/tflite_runtime-2.5.2-cp38-cp38-linux_x86_64.whl

COPY clothing-model.tflite lambda_function.py ./

CMD ["lambda_function.lambda_handler"]
