import pytest
from html_transpiler.html_transpiler import HTMLTranspiler
from pathlib import Path

@pytest.fixture
def valid_dsl_mapping_file_path():
    return Path(Path(__file__).parent.absolute(), "web-dsl-mapping.json")

@pytest.mark.parametrize("tokens", [
    None,
    [],
])
def test_empty_tokens(valid_dsl_mapping_file_path, tokens):
    with pytest.raises(Exception):
        transpiler = HTMLTranspiler(valid_dsl_mapping_file_path)
        transpiler.transpile(tokens)

def test_missing_variable_mapping_in_tokens(valid_dsl_mapping_file_path):
    invalid_tokens = ["header,", "{,,", "btn-inactive_asdf", "btn-inactive  ", "btn-inactive", "btn-active", "btn-inactive", "}"]
    with pytest.raises(Exception):
        transpiler = HTMLTranspiler(valid_dsl_mapping_file_path)
        transpiler.transpile(invalid_tokens)

def test_valid_tokens(valid_dsl_mapping_file_path):
    valid_tokens = ["header", "{", "btn-inactive", "btn-inactive", "btn-inactive", "btn-active", "btn-inactive", "}", "row", "{", "single", "{", "small-title", "text", "btn-red", "}", "}"]
    correct_output = '<html>\n  <header>\n    <meta charset="utf-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1">\n    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">\n<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">\n<style>\n.header{margin:20px 0}nav ul.nav-pills li{background-color:#333;border-radius:4px;margin-right:10px}.col-lg-3{width:24%;margin-right:1.333333%}.col-lg-6{width:49%;margin-right:2%}.col-lg-12,.col-lg-3,.col-lg-6{margin-bottom:20px;border-radius:6px;background-color:#f5f5f5;padding:20px}.row .col-lg-3:last-child,.row .col-lg-6:last-child{margin-right:0}footer{padding:20px 0;text-align:center;border-top:1px solid #bbb}\n</style>\n    <title>Scaffold</title>\n  </header>\n  <body>\n    <main class="container">\n      <div class="header clearfix">\n  <nav>\n    <ul class="nav nav-pills pull-left">\n      <li><a href="#">[]</a></li>\n<li><a href="#">[]</a></li>\n<li><a href="#">[]</a></li>\n<li class="active"><a href="#">[]</a></li>\n<li><a href="#">[]</a></li>\n\n    </ul>\n  </nav>\n</div>\n<div class="row"><div class="col-lg-12">\n<h4>[]</h4><p>[]</p>\n<a class="btn btn-danger" href="#" role="button">[]</a>\n</div>\n</div>\n\n      <footer class="footer">\n        <p>&copy; Tony Beltramelli 2017</p>\n      </footer>\n    </main>\n    <script src="js/jquery.min.js"></script>\n    <script src="js/bootstrap.min.js"></script>\n  </body>\n</html>\n'

    transpiler = HTMLTranspiler(valid_dsl_mapping_file_path)
    output = transpiler.transpile(valid_tokens)

    assert output == correct_output

def test_content_placeholder_insertion(valid_dsl_mapping_file_path):
    # Remember to set the random seed
    valid_tokens = ["header", "{", "btn-inactive", "btn-inactive", "btn-inactive", "btn-active", "btn-inactive", "}", "row", "{", "single", "{", "small-title", "text", "btn-red", "}", "}"]

    transpiler = HTMLTranspiler(valid_dsl_mapping_file_path)
    output = transpiler.transpile(valid_tokens, insert_random_text=True)
    print("s")
    assert len(output) > len(valid_tokens)