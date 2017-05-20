import argparse
import os
import subprocess
from bs4 import BeautifulSoup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert notebook to hugo mardown')
    parser.add_argument('--input_nb',   action='store', help='path to notebook')
    parser.add_argument('--hugo_dir',   action='store', help='path to hugo directory')
    parser.add_argument('--outfile',    action='store', help='relative path to hugo directory')

    args = parser.parse_args()

    if not os.path.isfile(args.input_nb):
        raise Exception('Can NOT open input notebook <{}>'.format(args.input_nb))

    meta_file = '{}.meta'.format(args.input_nb)
    if not os.path.isfile(meta_file):
        raise Exception('Can NOT open input notebook\'s meta-file <{}>'.format(meta_file))

    indir = os.path.dirname(args.input_nb)
    output = os.path.join(args.hugo_dir, args.outfile)
    outdir = os.path.dirname(output)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    cmd = 'jupyter nbconvert --to html --template basic {}'.format(args.input_nb)

    subprocess.check_output(cmd.split())
    out_html = '{}.html'.format(args.input_nb[:-len('.ipynb')])
    img_files = []
    out_imgs = []

    fname, fext = os.path.splitext(os.path.basename(output))

    with open(out_html, 'r') as f:
        body = f.read()

        # # replace link
        soup = BeautifulSoup(body, "html.parser")

        for a in soup.findAll('a'):
            if a['href'].startswith('./') and a['href'].endswith('.ipynb'):
                a['href'] = a['href'][:-len('.ipynb')] + '/'

        for a in soup.findAll('img'):
            img_files.append(os.path.normpath(os.path.join(indir, a['src'])))
            out_imgs.append(os.path.normpath(os.path.join(outdir, fname, a['src'])))

        body = soup.prettify()

    # done with html file, remove it now
    os.remove(out_html)

    if len(img_files) > 0:
        print ('\nStart copying images to hugo dir...\n-----------------------------------')
        for a, b in zip(img_files, out_imgs):
            if not os.path.isfile(a):
                continue
            bdir = os.path.dirname(b)
            if not os.path.isdir(bdir):
                os.makedirs(bdir)
            cmd = 'cp -r {} {}'.format(a, b)
            subprocess.check_output(cmd.split())
            print ('copied {} => {} done'.format(a, b))
        print ("\n")

    with open(meta_file, 'r') as f:
        meta = f.read()

    with open(output, 'w') as f:
        f.write('{}\n'.format(meta))
        f.write('{}'.format(body))

    print ('Convert notebook {} using meta {} ==> {}\n=================================\n'.format(args.input_nb, meta_file, output))