class Policy:
    def __init__(self):
        self.common = Common()


class Common:
    def error(self, msg="Common Error"):
        if not isinstance(msg, str):
            msg = "You must provide msg as a str"
        raise Exception(msg)


policy = Policy()
