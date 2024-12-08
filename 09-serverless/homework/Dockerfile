FROM  agrigorev/model-2024-hairstyle:v3

RUN pip install --no-cache-dir  --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl \
    && pip install --no-cache-dir pillow

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]