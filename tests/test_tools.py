from tools import retrieve_information

def test():
    res = retrieve_information("What is the hotel allowance in Japan?", "knowledge_base.txt")
    print("Found chunks:")
    for r in res:
        print("---")
        print(r)

if __name__ == "__main__":
    test()
