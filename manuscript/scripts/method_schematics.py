"""Scientific protocol schematics; all node positions and paths are illustrative."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, ORANGE, GREEN, RED, PURPLE = '#2478A6', '#C78336', '#558760', '#BB5364', '#87629C'
INK, GRAY = '#25333D', '#747F88'


def text(ax, x, y, s, size=9, **kw):
    ax.text(x, y, s, fontsize=size, color=kw.pop('color', INK),
            ha=kw.pop('ha', 'center'), va='center', **kw)


def box(ax, x, y, w, h, label='', color=BLUE, fill='#F1F6F9', size=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.015',
                              facecolor=fill,edgecolor=color,lw=1))
    if label: text(ax,x+w/2,y+h/2,label,size)


def arrow(ax, p, q, color=GRAY, style='->', lw=1.15):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle=style,mutation_scale=10,color=color,lw=lw))


def base(n, height):
    fig, axes=plt.subplots(n,1,figsize=(7.3,height))
    fig.subplots_adjust(left=.025,right=.975,bottom=.04,top=.98,hspace=.21)
    for ax in axes:
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    return fig,axes


def title(ax, label):
    text(ax,.015,.96,label,10.5,ha='left',fontweight='bold')


def comparison_design():
    fig,axes=base(3,6.0)
    ax=axes[0];title(ax,'A  Initial geometry and individual identity are different questions')
    text(ax,.015,.81,'Main cohort: 10 networks per dataset',8.5,ha='left')
    for center,label,phase in [(.14,'Initial weights (E0)',0),(.46,'Final weights (E20)',.33)]:
        box(ax,center-.112,.16,.224,.51,color='#CDD8DF',fill='#FAFCFD')
        angle=np.arange(10)*2*np.pi/10+phase
        radius=np.ones(10) if phase==0 else np.array([.70,1.10,.86,.58,.92,.76,1.08,.65,.84,1.00])
        points=np.column_stack([center+.078*radius*np.cos(angle),.42+.17*radius*np.sin(angle)])
        for i,j in [(0,3),(0,6),(2,7),(4,8)]:
            ax.plot(points[[i,j],0],points[[i,j],1],color='#CAD6DD',lw=.8,zorder=1)
        ax.scatter(points[:,0],points[:,1],s=25,c=[BLUE]*9+[ORANGE],zorder=3)
        text(ax,center,.075,label,8)
    arrow(ax,(.26,.43),(.34,.43))
    text(ax,.30,.61,'10 training\nepochs',7.5)
    box(ax,.64,.48,.335,.23,'RSA\n45 initial vs. 45 final distances',PURPLE,'#F5F2F8',8.4)
    box(ax,.64,.11,.335,.27,'Identity retrieval\nEach endpoint vs. 10 E0 candidates\nWeights or hidden CKA',BLUE,size=8.1)
    arrow(ax,(.58,.55),(.63,.59),PURPLE)
    arrow(ax,(.58,.31),(.63,.25),BLUE)

    ax=axes[1];title(ax,'B  Cross initialization with batch schedule')
    x0,y0,w,h=.105,.17,.29,.57;cw=w/5;ch=h/10
    for i in range(10):
        for j in range(5):
            fill='#E2EDF3' if i==4 else ('#F3E7D8' if j==2 else '#F5F7F8')
            ax.add_patch(Rectangle((x0+j*cw,y0+(9-i)*ch),cw,ch,facecolor=fill,edgecolor='white',lw=.7))
            ax.plot(x0+(j+.5)*cw,y0+(9-i+.5)*ch,'o',ms=2.3,color=BLUE if i==4 else (ORANGE if j==2 else '#A9B5BC'))
    for j in range(5):text(ax,x0+(j+.5)*cw,.80,str(j+1),7.5)
    text(ax,.25,.89,'Batch schedule',8)
    text(ax,.036,.46,'Initialization',8,rotation=90)
    for i in [0,4,9]:text(ax,.083,y0+(9-i+.5)*ch,str(i+1),7.5)
    text(ax,.25,.07,'50 runs per dataset; 150 total',8)
    box(ax,.51,.49,.46,.24,'Same initialization, different schedules\nIdentify origin among 10 candidates',BLUE,size=8.5)
    box(ax,.51,.13,.46,.24,'Same schedule, different initializations\nIdentify schedule among 5 candidates',ORANGE,'#FBF5ED',8.5)
    arrow(ax,(.405,.49),(.50,.61),BLUE)
    arrow(ax,(.405,.24),(.50,.25),ORANGE)

    ax=axes[2];title(ax,'C  Perturb weights and compare paired continuations')
    text(ax,.39,.81,'Same future batches and momentum protocol',8.5)
    ax.scatter([.08],[.43],s=48,color=INK,zorder=5)
    text(ax,.08,.65,'Source\nnetwork',8)
    t=np.linspace(0,1,100)
    x=.10+.54*t
    upper=.44+.12*np.sin(t*np.pi/2)
    lower=.29+.035*np.sin(t*np.pi)
    ax.plot(x,upper,color=BLUE,lw=2)
    ax.plot(x,lower,color=RED,lw=2)
    arrow(ax,(.08,.40),(.10,.30),RED)
    text(ax,.38,.65,'Unperturbed continuation',8,color=BLUE)
    text(ax,.39,.19,'Perturbed continuation',8,color=RED)
    ax.scatter([x[-1],x[-1]],[upper[-1],lower[-1]],s=30,color=[BLUE,RED])
    arrow(ax,(.68,lower[-1]),(.68,upper[-1]),INK,'<->')
    text(ax,.70,.425,r'$d_Q(s)$',8,ha='left')
    box(ax,.79,.26,.18,.34,'Separation\n'+r'$r(s) =$'+'\n'+r'$d_Q(s) / d_Q(0^+)$',PURPLE,'#F5F2F8',8)
    text(ax,.38,.055,'10 further epochs; accuracy measured separately',8)
    fig.text(.5,.009,'Protocol sketches; node positions and continuation paths are illustrative.',ha='center',fontsize=7.5,color=GRAY)
    return fig


def task(ax,x,y,w,label,color):
    box(ax,x,y,w,.115,label,color,{'A':'#E3EFF5','B':'#F8EBD9','C':'#E6F0E8'}.get(label[0],'#F3F4F5'),8)


def decoding_design():
    fig,axes=base(3,6.0)
    ax=axes[0];title(ax,'A  Paired histories share initial weights and image order')
    text(ax,.225,.79,'History experiment: 1 + 1 + 10 epochs',8.3)
    for y,labs in [(.53,['A','B','C: 10 epochs']),(.34,['B','A','C: 10 epochs'])]:
        task(ax,.035,y,.07,labs[0],BLUE if labs[0]=='A' else ORANGE)
        task(ax,.125,y,.07,labs[1],BLUE if labs[1]=='A' else ORANGE)
        task(ax,.215,y,.23,labs[2],GREEN)
        arrow(ax,(.106,y+.057),(.122,y+.057));arrow(ax,(.197,y+.057),(.212,y+.057))
    text(ax,.24,.19,'A: digit identity   B: labels +1   C: labels +3',7.6)
    text(ax,.24,.075,'All labels modulo 10',7.6,color=GRAY)
    text(ax,.75,.79,'Trajectory experiment: one or four pairs',8.3)
    for y,seq in [(.53,['AB','BA','AB','BA']),(.34,['BA','AB','BA','AB'])]:
        for i,lab in enumerate(seq):
            box(ax,.525+i*.079,y,.069,.115,lab,PURPLE,'#F5F2F8',8)
        task(ax,.855,y,.12,'C: 5 ep.',GREEN)
    text(ax,.75,.19,'Example: four complementary AB/BA pairs',7.6)
    text(ax,.75,.075,'One epoch per A/B phase; five final C epochs',7.6,color=GRAY)

    ax=axes[1];title(ax,'B  Split trained networks by initialization seed')
    text(ax,.015,.79,'200 seed pairs per cohort; both histories always stay together',8.5,ha='left')
    for x,w,col,head,sub in [(.025,.295,BLUE,'120 training seeds','240 networks\nFit preprocessing and decoder'),
                           (.355,.29,ORANGE,'40 validation seeds','80 networks\nSelect regularization'),
                           (.680,.295,GREEN,'40 test seeds','80 networks\nEvaluate held-out predictions')]:
        box(ax,x,.25,w,.42,color=col,fill='#FAFCFD')
        text(ax,x+w/2,.565,head,9,fontweight='bold',color=col)
        text(ax,x+w/2,.39,sub,8.3)
    arrow(ax,(.324,.46),(.351,.46));arrow(ax,(.649,.46),(.676,.46))
    text(ax,.5,.10,'Reference alignment, scaling, and PCA are fitted on training seeds only.',8.2)

    ax=axes[2];title(ax,'C  Read an event or predict an earlier state')
    box(ax,.025,.29,.22,.29,'Final weights only\nCanonicalize and align',BLUE,size=8.7)
    box(ax,.345,.51,.28,.23,'History classifier',PURPLE,'#F5F2F8',9)
    box(ax,.345,.13,.28,.23,'Ridge regression',ORANGE,'#FBF5ED',9)
    box(ax,.715,.51,.26,.23,'Earlier task order\nAB or BA',PURPLE,'#F5F2F8',8.8)
    box(ax,.715,.13,.26,.23,'Earlier position z(t)\n8 PCA coordinates',ORANGE,'#FBF5ED',8.8)
    arrow(ax,(.25,.49),(.337,.62),PURPLE);arrow(ax,(.25,.37),(.337,.25),ORANGE)
    arrow(ax,(.63,.62),(.707,.62),PURPLE);arrow(ax,(.63,.25),(.707,.25),ORANGE)
    text(ax,.5,.025,'Earlier checkpoints define regression targets; they are not decoder inputs.',8.2)
    return fig
