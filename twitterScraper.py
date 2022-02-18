import twint


def scrape(keyword, scope, fileName):
    c = twint.Config()

    # topic to scrape
    c.Search = [keyword]

    # number of tweets to scrape
    c.Limit = scope

    c.Store_csv = True
    c.Output = fileName

    twint.run.Search(c)

