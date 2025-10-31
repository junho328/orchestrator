
import os ,sys ,shutil 

def main ():
    if len (sys .argv )!=2 :
        print (f'usage: {sys.argv[0]} <path>')
        sys .exit (1 )
    root =sys .argv [1 ]

    for dirpath ,dirnames ,_ in os .walk (root ):
        for d in list (dirnames ):
            if d in ('.git','.svn','__pycache__'):
                path =os .path .join (dirpath ,d )
                try :
                    shutil .rmtree (path )
                    print (f'removed {path}')
                except Exception as e :
                    print (f'error removing {path}: {e}')
    print ('clean complete')

if __name__ =='__main__':
    main ()
