from minio import Minio


class MinioRepo:
    def __init__(self, settings):
        self.client = Minio(settings['minio_host'],
                       access_key=settings['access_key'],
                       secret_key=settings['secret_key'],
                       secure=settings['secure'])

    def get_global_model(self, model_id, bucket=''):
        if bucket == '':
            bucket = self.settings['bucket']
        try:
            data = self.client.get_object(bucket, model_id)
            return data.read()
        except Exception as e:
            raise Exception("Could not fetch data from bucket, {}".format(e))

    def get_global_model_list(self, bucket=''):
        objects_list = []
        if bucket == '':
            bucket = self.settings['bucket']
        try:
            objects = self.client.list_objects(bucket)
            for obj in objects:
                objects_list.append(obj.object_name)
            return objects_list
        except Exception as e:
            raise Exception("Could not fetch object list from bucket, {}".format(e))

    def get_object_list(self, bucket=''):
        objects_list = []
        if bucket == '':
            bucket = self.settings['bucket']
        try:
            objects = self.client.list_objects(bucket)
            for obj in objects:
                objects_list.append(obj)
            return objects_list

        except Exception as e:
            raise Exception("Could not fetch object list from bucket, {}".format(e))