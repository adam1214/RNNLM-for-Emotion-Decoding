import joblib

def split_dialog(dialogs):
  """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
     See eq (5) in the paper.
  Arg:
    dialogs: dict, for example, utterances of two speakers in dialog_01: 
            {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
  Return:
    spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
            {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
             dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
  """

  spk_dialogs = {}
  for dialog_id in dialogs.keys():
    #spk_dialogs[dialog_id+'_M'] = []
    #spk_dialogs[dialog_id+'_F'] = []
    for utt_id in dialogs[dialog_id]:
      if utt_id[-4] == 'M':
        if spk_dialogs.get(dialog_id+'_M') == None:
            spk_dialogs[dialog_id+'_M'] = []
        spk_dialogs[dialog_id+'_M'].append(utt_id)
      elif utt_id[-4] == 'F':
        if spk_dialogs.get(dialog_id+'_F') == None:
            spk_dialogs[dialog_id+'_F'] = []
        spk_dialogs[dialog_id+'_F'].append(utt_id)

  return spk_dialogs

dialogs = joblib.load('dialog_rearrange_4emo_iemocap.pkl')
spk_dialogs = split_dialog(dialogs)
emo_all = joblib.load('emo_all_iemocap.pkl')
emo_shift_all = joblib.load('4emo_shift_all_rearrange.pkl')

total_cnt_after_shift = 0
shift_after_shift = 0
for dia in spk_dialogs:
    for i, utt in enumerate(spk_dialogs[dia]):
        if i+1 < len(spk_dialogs[dia]):
            next_utt = spk_dialogs[dia][i+1]
            if emo_shift_all[utt] == 1.0:
                total_cnt_after_shift += 1
                if emo_all[utt] != emo_all[next_utt]:
                    shift_after_shift += 1

print('data points of inertia after shift:', 100-round(100*shift_after_shift/total_cnt_after_shift,2), '%')
print('data points of shift after shift:', round(100*shift_after_shift/total_cnt_after_shift,2), '%')