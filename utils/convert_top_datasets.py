import pandas as pd
import numpy as np
import awkward0 as ak0
import math
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser('Convert top benchmark h5 datasets to awkd')
parser.add_argument('-c', '--condition', required=True, help='Create dataset for training/test. Set to \'train\' or \'test\'.')
parser.add_argument('--max-event-size', default=100000, help='Maximum event size per output file.')
args = parser.parse_args()

if args.condition == 'train':
    inputfiles, outputfile = ['train.h5'], 'top_train'
elif args.condition == 'val':
    inputfiles, outputfile = ['val.h5'], 'top_val'
elif args.condition == 'test':
    inputfiles, outputfile = ['test.h5'], 'top_test'
else:
    raise RuntimeError('Invalid --condition option')

def store_file(res_array, outname):
    print('Saving to file', outname, '...')
    ak0.save(outname, ak0.fromiter(res_array), mode='w')
 
def main():
    varlist_2d = ['E', 'PX', 'PY', 'PZ']
    varlist_1d = ['truthE', 'truthPX', 'truthPY', 'truthPZ', 'ttv', 'is_signal_new']
    varlist_2d_new = ['E_log', 'P', 'P_log', 'Etarel', 'Phirel']
    varlist_1d_new = ['E_tot', 'PX_tot', 'PY_tot', 'PZ_tot', 'P_tot', 'Eta_tot', 'Phi_tot']
    idx, ifile = 0, 0
    res_array = []
    for filename in inputfiles:
        print('Reading table from:', filename, '...')
        with pd.HDFStore(f'samples/{filename}') as store:
            df = store.select('table')

        print('Processing events ...')
        isfirst = True
        for origIdx, row in tqdm(df.iterrows()):
            # First initiate 2d arrays
            res = {k:[] for k in varlist_2d + varlist_2d_new}
            nPart = 0
            for ipar in range(200):
                if row[f'E_{ipar}'] == 0.:
                    break
                for k in ['E', 'PX', 'PY', 'PZ']:
                    res[k].append(row[f'{k}_{ipar}'])
                res['E_log'].append(math.log(row[f'E_{ipar}']))
                res['P'].append(math.sqrt(row[f'PX_{ipar}']**2 + row[f'PY_{ipar}']**2 + row[f'PZ_{ipar}']**2))
                res['P_log'].append(math.log(res['P'][-1]))
                nPart += 1
            
            # Fill 1d arrays
            for k in varlist_1d:
                res[k] = row[k]
            res['E_tot'] = sum(res['E'])
            res['PX_tot'] = sum(res['PX'])
            res['PY_tot'] = sum(res['PY'])
            res['PZ_tot'] = sum(res['PZ'])
            res['P_tot'] = math.sqrt(res['PX_tot']**2 + res['PY_tot']**2 + res['PZ_tot']**2)
            res['Eta_tot'] = math.atanh(res['PZ_tot'] / res['P_tot'])
            res['Phi_tot'] = math.atan(res['PY_tot'] / res['PX_tot'])
            res['nPart'] = nPart
            res['origIdx'] = origIdx
            res['idx'] = idx
            
            # Calculate new 2d arrays
            for ipar in range(nPart):
                Eta = math.atanh(res['PZ'][ipar] / res['P'][ipar])
                Phi = math.atan(res['PY'][ipar] / res['PX'][ipar])
                res['Etarel'].append(np.sign(res['Eta_tot']) * (Eta - res['Eta_tot']))
                res['Phirel'].append(Phi - res['Phi_tot'] - 2 * math.pi * math.floor((Phi - res['Phi_tot']) / (2 * math.pi) + 0.5))

            # Store per event result
            res_array.append(res)
            idx += 1
                        
            if isfirst:
                print(res)
                isfirst = False

            if idx >= args.max_event_size:
                store_file(res_array, f'{outputfile}_{ifile}.awkd')
                del res_array
                res_array = []
                ifile += 1
                idx = 0

    # Save rest of events before finishing
    store_file(res_array, f'{outputfile}_{ifile}.awkd')


if __name__ == '__main__':
    main()
