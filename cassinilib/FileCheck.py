def FileCheck(fn):
    ''' Check if a file exists and return a boolean '''
    try:
      open(fn, "r")
      return 1
    except IOError:
      print("Error: File does not appear to exist.")
      return 0
