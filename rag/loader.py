import os



def load_documents(path):


    documents=[]


    for root,dirs,files in os.walk(path):


        for file in files:


            file_path=os.path.join(
                root,
                file
            )


            if file.endswith(
                (".txt",".md")
            ):


                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:


                    text=f.read()



                documents.append(

                    {

                    "content":text,

                    "source":file_path

                    }

                )


    return documents
