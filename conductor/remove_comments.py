
import argparse 
import ast 
import io 
import tokenize 
import sys 


def find_docstring_lines (src ):
    tree =ast .parse (src )
    doc_lines =set ()

    mod_doc =ast .get_docstring (tree ,clean =False )
    if mod_doc :
        first =tree .body [0 ]
        if (
        isinstance (first ,ast .Expr )
        and isinstance (first .value ,ast .Constant )
        and isinstance (first .value .value ,str )
        ):
            doc_lines .add (first .lineno )

    for node in ast .walk (tree ):
        if isinstance (
        node ,
        (
        ast .FunctionDef ,
        ast .AsyncFunctionDef ,
        ast .ClassDef ,
        ),
        ):
            if node .body and isinstance (node .body [0 ],ast .Expr ):
                val =node .body [0 ].value 
                if isinstance (val ,ast .Constant )and isinstance (val .value ,str ):
                    doc_lines .add (node .body [0 ].lineno )
    return doc_lines 


def remove_comments_and_doc (src ):
    doc_lines =find_docstring_lines (src )
    out_tokens =[]
    gen =tokenize .generate_tokens (io .StringIO (src ).readline )
    for tok_type ,tok_str ,start ,end ,line in gen :
        lineno ,_ =start 

        if tok_type ==tokenize .COMMENT :
            continue 

        if tok_type ==tokenize .STRING and lineno in doc_lines :
            continue 
        out_tokens .append ((tok_type ,tok_str ))
    return tokenize .untokenize (out_tokens )

def main ():
    p =argparse .ArgumentParser (
    description ='remove comments and docstrings from a Python file'
    )
    p .add_argument ('infile',help ='source file path')
    p .add_argument (
    '-o',
    '--out',
    help ='output file path (default: stdout)',
    )
    args =p .parse_args ()
    data =open (args .infile ,encoding ='utf-8').read ()
    cleaned =remove_comments_and_doc (data )
    if args .out :
        open (args .out ,'w',encoding ='utf-8').write (cleaned )
    else :
        sys .stdout .write (cleaned )

if __name__ =='__main__':
    main ()
