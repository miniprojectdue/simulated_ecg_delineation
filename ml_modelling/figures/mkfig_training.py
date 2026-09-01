
import json, re, numpy as np, matplotlib.pyplot as plt
import figstyle as F
F.apply()

STEP = re.compile(r"epoch\s+(\d+)\s+step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+ce\s+([\d.]+)\s+dice\s+([\d.]+)")

def per_step(run):
    S=L=C=D=None; xs=[]; loss=[]; ce=[]; dice=[]; T=None
    for line in open('ml_modelling/results/%s/train_log.txt' % run):
        m=STEP.search(line)
        if not m: continue
        e,s,t=int(m.group(1)),int(m.group(2)),int(m.group(3)); T=t
        xs.append((e-1)*t+s); loss.append(float(m.group(4))); ce.append(float(m.group(5))); dice.append(float(m.group(6)))
    return np.array(xs), np.array(loss), np.array(ce), np.array(dice), T

def per_epoch(run):
    h=json.load(open('ml_modelling/results/%s/history.json' % run))
    ep=np.array([r['epoch'] for r in h])
    val=np.array([r.get('val_loss',np.nan) for r in h])
    tot=np.array([r['train']['loss'] for r in h]); ce=np.array([r['train']['ce'] for r in h]); dice=np.array([r['train']['dice'] for r in h])
    return ep,tot,ce,dice,val

fig,axes=plt.subplots(1,2,figsize=(F.TEXTWIDTH,3.0))

# stage one, step resolution
ax=axes[0]
xs,loss,ce,dice,T=per_step('pretrain_toffset_fix')
ep,_,_,_,val=per_epoch('pretrain_toffset_fix')
xk=xs/1000.0
ax.plot(xk,loss,color=F.INK,lw=1.0,label='training loss',zorder=5)
ax.plot(xk,ce,color=F.BLUE,lw=0.8,label='cross-entropy',zorder=4)
ax.plot(xk,dice,color=F.GREEN,lw=0.8,label='soft Dice',zorder=4)
ax.plot(ep*T/1000.0,val,color=F.RED,lw=1.1,ls='--',dashes=(4,2.4),marker='o',ms=2.4,label='validation loss',zorder=6)
ax.set_title('stage one, pretraining',loc='left',fontsize=8.8,color=F.INK,pad=6)
ax.set_xlabel('training step (thousands)',labelpad=1); ax.set_ylabel('loss',labelpad=2)
ax.set_ylim(bottom=0); ax.set_xlim(0,xk.max()); F.despine(ax)

# stage two, per epoch
ax=axes[1]
ep,tot,ce,dice,val=per_epoch('finetune_toffset_fix')
ax.plot(ep,tot,color=F.INK,lw=1.4,label='training loss',zorder=5)
ax.plot(ep,ce,color=F.BLUE,lw=1.1,label='cross-entropy',zorder=4)
ax.plot(ep,dice,color=F.GREEN,lw=1.1,label='soft Dice',zorder=4)
ax.plot(ep,val,color=F.RED,lw=1.1,ls='--',dashes=(4,2.4),marker='o',ms=2.6,label='validation loss',zorder=6)
ax.set_title('stage two, fine-tuning',loc='left',fontsize=8.8,color=F.INK,pad=6)
ax.set_xlabel('epoch',labelpad=1)
ax.set_ylim(bottom=0); ax.set_xlim(ep.min(),ep.max()); F.despine(ax)
ax.legend(loc='upper right',fontsize=7.4,borderaxespad=0.3,handlelength=1.6,labelspacing=0.3)

fig.subplots_adjust(wspace=0.22)
fig.savefig('Dissertation/images/fig_training_curves.pdf')
fig.savefig('Dissertation/images/fig_training_curves.png',dpi=400)
print('written; stage1 %d step-points over %d steps/epoch, stage2 %d epochs'%(len(xs),T,len(ep)))
