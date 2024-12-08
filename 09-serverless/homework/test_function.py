import lambda_function


url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
result = lambda_function.predict(url)
print(result)