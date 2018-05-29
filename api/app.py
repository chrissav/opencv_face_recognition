from flask import Flask, request, jsonify, Response
import json
import sqlite3 as sqlite
app = Flask(__name__)

import cv2
import numpy as np 
import sqlite3
import os
import argparse
from PIL import Image


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d
    
@app.route('/', methods=['GET'])
def home():
    return """<h1>Facial Recognition Testing</h1>
    <p>A prototype API for testing facial recognition detection with different datasets and algorithms</p>
    """


@app.route('/api/v1/resources/users/all', methods=['GET'])
def get_all_users():
    conn = sqlite.connect('../data/database.db')
    conn.row_factory = dict_factory
    cur = conn.cursor()
    all_users = cur.execute("SELECT * FROM users;").fetchall()
    return jsonify(all_users)

@app.route("/api/v1/resources/books", methods=['GET'])
def api_filter():
    query_parameters = request.args
    
    id = query_parameters.get('id')
    published = query_parameters.get('published')
    author = query_parameters.get('author')
    
    
    query = "SELECT * FROM books WHERE"
    to_filter = []
    
    if id:
        query += ' id=? AND'
        to_filter.append(id)
    if published:
        query += ' published=? AND'
        to_filter.append(published)
    if author:
        query += ' author=? AND'
        to_filter.append(author)
    if not (id or published or author):
        return page_not_found(404)
    
    query = query[:-4] + ';'
    
    conn = sqlite.connect('../data/books.db')
    conn.row_factory = dict_factory
    cur = conn.cursor()
    
    results = cur.execute(query, to_filter).fetchall()
    
    return jsonify(results)
    

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
