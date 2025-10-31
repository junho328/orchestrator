

import re 
from math import isclose 

import regex 

from sympy import N ,simplify 
from sympy .parsing .latex import parse_latex 
from sympy .parsing .sympy_parser import parse_expr 
from word2number import w2n 


def convert_word_number (text :str )->str :
    try :
        text =str (w2n .word_to_num (text ))
    except Exception :
        pass 
    return text 


def _fix_fracs (string ):
    substrs =string .split ("\\frac")
    new_str =substrs [0 ]
    if len (substrs )>1 :
        substrs =substrs [1 :]
        for substr in substrs :
            new_str +="\\frac"
            if len (substr )>0 and substr [0 ]=="{":
                new_str +=substr 
            else :
                try :
                    assert len (substr )>=2 
                except Exception :
                    return string 
                a =substr [0 ]
                b =substr [1 ]
                if b !="{":
                    if len (substr )>2 :
                        post_substr =substr [2 :]
                        new_str +="{"+a +"}{"+b +"}"+post_substr 
                    else :
                        new_str +="{"+a +"}{"+b +"}"
                else :
                    if len (substr )>2 :
                        post_substr =substr [2 :]
                        new_str +="{"+a +"}"+b +post_substr 
                    else :
                        new_str +="{"+a +"}"+b 
    string =new_str 
    return string 


def _fix_a_slash_b (string ):
    if len (string .split ("/"))!=2 :
        return string 
    a =string .split ("/")[0 ]
    b =string .split ("/")[1 ]
    try :
        if "sqrt"not in a :
            a =int (a )
        if "sqrt"not in b :
            b =int (b )
        assert string =="{}/{}".format (a ,b )
        new_string ="\\frac{"+str (a )+"}{"+str (b )+"}"
        return new_string 
    except Exception :
        return string 


def _fix_sqrt (string ):
    _string =re .sub (r"\\sqrt(\w+)",r"\\sqrt{\1}",string )
    return _string 


def strip_answer_string (string ):

    string =str (string ).strip ()

    string =string .replace ("\n","")


    string =string .rstrip (".")



    string =string .replace ("\\!","")




    string =re .sub (r"\\begin\{array\}\{.*?\}",r"\\begin{pmatrix}",string )
    string =re .sub (r"\\end\{array\}",r"\\end{pmatrix}",string )
    string =string .replace ("bmatrix","pmatrix")


    string =string .replace ("tfrac","frac")
    string =string .replace ("dfrac","frac")
    string =(
    string .replace ("\\neq","\\ne")
    .replace ("\\leq","\\le")
    .replace ("\\geq","\\ge")
    )


    string =string .replace ("\\left","")
    string =string .replace ("\\right","")
    string =string .replace ("\\{","{")
    string =string .replace ("\\}","}")


    def replace_match (match ):
        word =match .group (1 ).lower ()
        if convert_word_number (word )==word :
            return match .group (0 )
        else :
            return convert_word_number (word )

    string =re .sub (r"\\text\{([a-zA-Z]+)\}",replace_match ,string )


    string =re .sub (r"(cm|inches)\}\^2",r"\1}",string )


    _string =re .sub (r"\\text{.*?}$","",string ).strip ()
    if _string !=""and _string !=string :

        string =_string 


    string =string .replace ("^{\\circ}","")
    string =string .replace ("^\\circ","")


    string =string .replace ("\\$","")
    string =string .replace ("$","")
    string =string .replace ("\\(","").replace ("\\)","")


    string =convert_word_number (string )


    string =re .sub (r"\\text\{(.*?)\}",r"\1",string )
    for key in ["x=","y=","z=","x\\in","y\\in","z\\in","x\\to","y\\to","z\\to"]:
        string =string .replace (key ,"")
    string =string .replace ("\\emptyset",r"{}")
    string =string .replace ("(-\\infty,\\infty)","\\mathbb{R}")


    string =string .replace ("\\%","")
    string =string .replace ("\%","")
    string =string .replace ("%","")


    string =string .replace (" ."," 0.")
    string =string .replace ("{.","{0.")



    if (
    string .startswith ("{")
    and string .endswith ("}")
    and string .isalnum ()
    or string .startswith ("(")
    and string .endswith (")")
    and string .isalnum ()
    or string .startswith ("[")
    and string .endswith ("]")
    and string .isalnum ()
    ):
        string =string [1 :-1 ]


    string =string .replace ("infinity","\\infty")
    if "\\infty"not in string :
        string =string .replace ("inf","\\infty")
    string =string .replace ("+\\inity","\\infty")


    string =string .replace ("and","")
    string =string .replace ("\\mathbf","")


    string =re .sub (r"\\mbox{.*?}","",string )


    string .replace ("'","")
    string .replace ('"',"")


    if "j"in string and "i"not in string :
        string =string .replace ("j","i")


    string =re .sub (r"(\d+)\.0*([^\d])",r"\1\2",string )
    string =re .sub (r"(\d+)\.0*$",r"\1",string )


    if len (string )==0 :
        return string 
    if string [0 ]==".":
        string ="0"+string 


    if len (string .split ("="))==2 :
        if len (string .split ("=")[0 ])<=2 :
            string =string .split ("=")[1 ]

    string =_fix_sqrt (string )
    string =string .replace (" ","")


    string =_fix_fracs (string )


    string =_fix_a_slash_b (string )


    string =re .sub (r"\\(?=\-?\d+(\\|\)|,|\]|$))","",string )


    string =re .sub (r"thgrade$","",string )


    if re .fullmatch (r"(\s*-?\d+\s*,)*\s*-?\d+\s*",string ):

        try :
            integer_list =list (map (int ,string .split (",")))
        except Exception :
            integer_list =list (map (int ,"-1,-1".split (",")))


        sorted_list =sorted (integer_list )


        string =",".join (map (str ,sorted_list ))

    return string 


def extract_answer (pred_str ,use_last_number =True ):

    pred_str =pred_str .replace ("\u043a\u0438","")
    if "final answer is $"in pred_str and "$. I hope"in pred_str :

        tmp =pred_str .split ("final answer is $",1 )[1 ]
        pred =tmp .split ("$. I hope",1 )[0 ].strip ()
    elif "boxed"in pred_str :
        ans =pred_str .split ("boxed")[-1 ]
        if len (ans )==0 :
            return ""
        elif ans [0 ]=="{":
            stack =1 
            a =""
            for c in ans [1 :]:
                if c =="{":
                    stack +=1 
                    a +=c 
                elif c =="}":
                    stack -=1 
                    if stack ==0 :
                        break 
                    a +=c 
                else :
                    a +=c 
        else :
            a =ans .split ("$")[0 ].strip ()
        pred =a 
    elif "he answer is"in pred_str :
        pred =pred_str .split ("he answer is")[-1 ].strip ()
    elif "final answer is"in pred_str :
        pred =pred_str .split ("final answer is")[-1 ].strip ()
    elif "答案是"in pred_str :

        pred =pred_str .split ("答案是")[1 ].strip ().split ("\n\n")[0 ].strip ()
    else :
        if use_last_number :
            pattern ="-?\d*\.?\d+"
            pred =re .findall (pattern ,pred_str .replace (",",""))
            if len (pred )>=1 :
                pred =pred [-1 ]
            else :
                pred =""
        else :
            pred =""



    pred =re .sub (r"\n\s*","",pred )
    if pred !=""and pred [0 ]==":":
        pred =pred [1 :]
    if pred !=""and pred [-1 ]==".":
        pred =pred [:-1 ]
    if pred !=""and pred [-1 ]=="/":
        pred =pred [:-1 ]
    pred =strip_answer_string (pred )
    return pred 


def get_multiple_choice_answer (pred :str ):

    tmp =re .findall (r"\b(A|B|C|D)\b",pred .upper ())
    if tmp :
        pred =tmp 
    else :
        pred =[pred .strip ().strip (".")]

    if len (pred )==0 :
        pred =""
    else :
        pred =pred [-1 ]


    pred =pred .rstrip (".").rstrip ("/")

    return pred 


def mmlu_pro_extract_answer (text ):
    pattern =r"answer is \(?([A-J])\)?"
    match =re .search (pattern ,text )
    if match :
        return match .group (1 )
    else :

        match =re .search (r".*[aA]nswer:\s*([A-J])",text )
        if match :
            return match .group (1 )
        else :

            pattern =r"\b[A-J]\b(?!.*\b[A-J]\b)"
            match =re .search (pattern ,text ,re .DOTALL )
            if match :
                return match .group (0 )


def choice_answer_clean (pred :str ):

    pred =pred .strip ("\n").rstrip (".").rstrip ("/").strip (" ").lstrip (":")

    tmp =re .findall (r"\b(A|B|C|D|E)\b",pred .upper ())
    if tmp :
        pred =tmp 
    else :
        pred =[pred .strip ().strip (".")]
    pred =pred [-1 ]

    pred =pred .rstrip (".").rstrip ("/")
    return pred 


def parse_digits (num ):
    num =regex .sub (",","",str (num ))
    try :
        return float (num )
    except Exception :
        if num .endswith ("%"):
            num =num [:-1 ]
            if num .endswith ("\\"):
                num =num [:-1 ]
            try :
                return float (num )/100 
            except Exception :
                pass 
    return None 