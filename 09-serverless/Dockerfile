FROM public.ecr.aws/lambda/python:3.10

RUN pip install --no-cache-dir keras-image-helper \
    && pip install --no-cache-dir --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime \
    && pip uninstall cryptography \
    && pip install --no-cache-dir cryptography==3.4.8

COPY clothing-model.tflite lambda_function.py ./

CMD ["lambda_function.lambda_handler"]
