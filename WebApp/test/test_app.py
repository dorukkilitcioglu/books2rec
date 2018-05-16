import sys
import pytest

# File to test
sys.path.append('./')
import app

def test_globals_startup():
    assert app.books is None
    assert app.bookid_to_title is None
    assert app.title_to_bookid is None
    assert app.mapper_id is None
    assert app.item_matrix is None
    assert app.qi is None
    assert app.top_recs_each_book_item_matrix is None
    assert app.top_recs_each_book_feature_matrix is None
    assert app.books is None
    assert app.titles is None

def test_load_books():
    app.load_books()
    assert len(app.titles) == 10000
    assert len(app.books.index) == 10000

def test_titles_mappers():
    app.load_title_mappers()
    assert app.bookid_to_title['1'] == 'The Hunger Games (The Hunger Games, #1)'
    assert app.title_to_bookid['The Hunger Games (The Hunger Games, #1)'] == '1'