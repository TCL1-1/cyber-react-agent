import requests



def web_search(query):


    url="https://duckduckgo.com/html/"


    response=requests.get(

        url,

        params={
            "q":query
        }

    )


    return response.text[:3000]
