import argparse
import os
import subprocess
from bs4 import BeautifulSoup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert notebook to hugo mardown')
    parser.add_argument('--input_nb',   action='store', help='path to notebook')
    parser.add_argument('--output',     action='store', help='path to output')

    args = parser.parse_args()

    if not os.path.isfile(args.input_nb):
        raise Exception('Can NOT open input notebook <{}>'.format(args.input_nb))

    meta_file = '{}.meta'.format(args.input_nb)
    if not os.path.isfile(meta_file):
        raise Exception('Can NOT open input notebook\'s meta-file <{}>'.format(meta_file))

    dir = os.path.dirname(args.output)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    cmd = 'jupyter nbconvert --to html --template basic {}'.format(args.input_nb)

    subprocess.check_output(cmd.split())
    output = '{}.html'.format(args.input_nb[:-len('.ipynb')])

    with open(output, 'r') as f:
        body = f.read()

        # # replace link
        soup = BeautifulSoup(body, "html.parser")

        for a in soup.findAll('a'):
            if a['href'].startswith('./') and a['href'].endswith('.ipynb'):
                a['href'] = a['href'][:-len('.ipynb')] + '/'

        for a in soup.findAll('img'):
            if a['src'].startswith('../assets'):
                a['src'] = a['href'][:-len('.ipynb')] + '/'

        body = soup.prettify()

    with open(meta_file, 'r') as f:
        meta = f.read()

    with open(args.output, 'w') as f:
        f.write('{}\n'.format(meta))
        f.write('{}'.format(body))

    print ('Convert notebook {} using meta {} ==> {}\n'.format(args.input_nb, meta_file, args.output))