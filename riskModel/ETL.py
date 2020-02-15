# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: ETL.py
# @Software: PyCharm

"""
connect database
"""

from pymongo import MongoClient
import pymysql


class DbRead(object):
    """从数据库中读取"""

    def __init__(self, host, port, user, password, database):
        """
        数据库连接信息
        :param host:str,IP
        :param port: int 端口
        :param user: str 用户名
        :param password: str 密码
        :param database: str or object 数据库名
        """
        self.__host = host
        self.__port = port
        self.__user = user
        self.__password = password
        self.__database = database

    def read_mysql(self, sql):
        """
        从mysql中获取数据
        :param sql: str, 执行的sql语句
        :return: data,iterable 数据生成器
        """
        dbConfig = {
            "host": self.__host,
            "port": self.__port,
            "user": self.__user,
            "password": self.__password,
            "db": self.__database,
            "cursorclass": pymysql.cursors.DictCursor,
            "charset": "utf8"
        }

        dbMysql = pymysql.connect(**dbConfig)
        cursor = dbMysql.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        dbMysql.close()
        return data

    def read_mongodb(self, collection, findCondition):
        """
        从mongoDB中获取数据
        :param collection: object,表名
        :param findCondition: dict 查询条件
        :return: data iterable 数据生成器
        """
        conn = MongoClient(host=self.__host, port=self.__port)
        dbMongo = conn.get_database(name=self.__database)
        dbMongo.authenticate(self.__user, self.__password)
        col = dbMongo.get_collection(collection)
        data = col.find(findCondition)
        conn.close()
        return data
