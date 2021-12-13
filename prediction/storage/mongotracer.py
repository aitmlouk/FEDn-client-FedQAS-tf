import pymongo


class MongoTracer:
    def __init__(self, host, network_id):
        try:
            mc = pymongo.MongoClient(host)
            # This is so that we check that the connection is live
            mc.server_info()
            self.mdb = mc[network_id]
            # self.mdb = connect_to_mongodb(host, network_id)
            self.messages = self.mdb['fedqas.messages']
        except Exception as e:
            print("FAILED TO CONNECT TO MONGO, {}".format(e), flush=True)
            self.status = None
            raise

    def set_prediction_info(self, global_model, context, question, answer, date_time):
        self.messages.update({'key': 'prediction'}, {'$push': {'global_model': global_model}}, True)
        self.messages.update({'key': 'prediction'}, {'$push': {'context': context}}, True)
        self.messages.update({'key': 'prediction'}, {'$push': {'question': question}}, True)
        self.messages.update({'key': 'prediction'}, {'$push': {'answer': answer}}, True)
        self.messages.update({'key': 'prediction'}, {'$push': {'date_time': date_time}}, True)
