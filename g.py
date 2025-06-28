import certifi
import ssl
import urllib.request

url = "https://twitter.com"
context = ssl.create_default_context(cafile=certifi.where())
response = urllib.request.urlopen(url, context=context)
print(response.status)
