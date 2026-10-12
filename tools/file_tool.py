def read_file(path):


    try:


        with open(path,"r",encoding="utf-8") as f:

            return f.read()

    except Exception as e:

        return str(e)
