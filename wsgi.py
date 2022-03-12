from main import app as app
if __name__ == "main":
  app.run(port=8888, debug=True, threaded = True)