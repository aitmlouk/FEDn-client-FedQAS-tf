# from flask import Flask, request
# from flask_restful import Api, Resource
#
# # instantiate Flask Rest Api
# app = Flask(__name__)
# api = Api(app)
#
#
# # Create class for Api Resource
# class Records(Resource):
#
#     # def get(self):
#     #     context = None
#     #     question = None
#     #     # get request that returns the JSON format for API request
#     #     return {"JSON data format": {"context": context,
#     #                                  "question": question,
#     #                                  }
#     #             }, 200
#
#     def post(self):
#         context = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program " \
#             "carried out by the National Aeronautics and Space Administration(NASA), which accomplished landing the " \
#             "first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration " \
#             "as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, " \
#             "Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and " \
#             "returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, " \
#             "address to Congress. Project Mercury was followed by the two-man Project Gemini. The first manned flight " \
#             "of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two man Gemini program " \
#             "which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel " \
#             "techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets " \
#             "as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, " \
#             "which consisted of Skylab, a space station that supported three manned missions in 1973-74, " \
#             "and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975. "
#
#         # post request
#         # make model and X_train global variables
#         # it gets patient's record and returns the ML model's prediction
#         data = request.get_json()
#         try:
#             context = str(context)
#             question = str(data["question"])
#
#             # get corresponding value from the diagnosis dictionary (using the model prediction as the key)
#             result = context + question
#             return {'Answer': result}, 200
#         except:
#             # if client sends the wrong request or data type then return correct format
#             return {'Error! Please use this JSON format': {#"context": "your context here...",
#                                                            "question": "your question here!",
#                                                            }}, 500
#
#
# api.add_resource(Records, '/api')
# app.run(port=5000, debug=True)
#
# # call API via curl
# # curl -d '{"context": 10, "question": 66}' -H "Content-Type: application/json" -X POST http://localhost:5000/api
