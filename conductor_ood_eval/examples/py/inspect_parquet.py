import pandas as pd 
import glob 


details_files =glob .glob ("path/to/details/file.parquet")
latest_file =details_files [0 ]


df =pd .read_parquet (latest_file )


breakpoint ()
wrong_answers =df [df ['gold']!=df ['prediction']]

print (f"Total samples: {len(df)}")
print (f"Wrong answers: {len(wrong_answers)}")
print (f"Accuracy: {(len(df) - len(wrong_answers)) / len(df) * 100:.1f}%")
print ("\n"+"="*80 )


for idx ,row in wrong_answers .iterrows ():
    print (f"\n--- WRONG ANSWER #{idx+1} ---")
    print (f"Problem: {row['query']}")
    print (f"Your answer: {row['prediction']}")
    print (f"Correct answer: {row['gold']}")
    print ("-"*40 )