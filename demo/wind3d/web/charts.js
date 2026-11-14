const $=id=>document.getElementById(id);
const colors={norm:'#e7f3fa',x:'#f58b83',y:'#69dfbd',z:'#79b9ff'};
let history=null,etag='',timer,controller,hover=null;
const canvases={force:$('force-chart'),moment:$('moment-chart')};
const keys=()=>$('history-components').value==='norm'?['norm']:['x','y','z'];
const value=(kind,key,i)=>{
  const offset=i*6+(kind==='force'?0:3),values=history.values;
  const raw=key==='norm'?Math.hypot(values[offset],values[offset+1],values[offset+2]):values[offset+['x','y','z'].indexOf(key)];
  return raw-($('history-demean').checked?history.meta.statistics[kind][key].mean:0);
};
const number=v=>Number(v.toPrecision(5)).toString();
function draw(kind){
  const canvas=canvases[kind],width=canvas.clientWidth,height=240,ratio=Math.min(devicePixelRatio,2);
  canvas.width=Math.round(width*ratio);canvas.height=height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
  if(!history){ctx.fillStyle='#8399ab';ctx.font='12px sans-serif';ctx.fillText('等待后端逐步载荷记录…',16,35);return;}
  const {meta,times}=history,series=keys(),left=74,right=width-18,top=18,bottom=202;
  let low=Infinity,high=-Infinity;
  for(const key of series){const stat=meta.statistics[kind][key],offset=$('history-demean').checked?stat.mean:0;low=Math.min(low,stat.min-offset);high=Math.max(high,stat.max-offset);}
  const padding=Math.max((high-low)*.08,.001);low-=padding;high+=padding;
  const start=times[0],end=times[times.length-1],span=end-start||Math.max(meta.dt_max,1e-6);
  const x=t=>left+(t-start)/span*(right-left),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
  ctx.font='10px monospace';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const yy=top+(bottom-top)*i/4,xx=left+(right-left)*i/4;
    ctx.strokeStyle='#243746';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(right,yy);ctx.stroke();
    ctx.fillStyle='#9db2c2';ctx.textAlign='right';ctx.fillText(number(high-(high-low)*i/4),left-8,yy+3);
    ctx.textAlign='center';ctx.fillText((start+span*i/4).toFixed(span<.05?5:4),xx,bottom+18);
  }
  ctx.textAlign='right';ctx.fillText('物理时间 t / s',right,height-3);
  ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
  ctx.setLineDash([3,4]);ctx.strokeStyle='#d0ab65';
  for(const event of meta.controls_events){if(event.time<=start||event.time>end)continue;ctx.beginPath();ctx.moveTo(x(event.time),top);ctx.lineTo(x(event.time),bottom);ctx.stroke();}
  ctx.setLineDash([]);
  for(const key of series){
    ctx.strokeStyle=colors[key];ctx.lineWidth=1.2;ctx.beginPath();
    for(let i=0;i<times.length;i++){const xx=x(times[i]),yy=y(value(kind,key,i));if(i===0||history.steps[i]!==history.steps[i-1]+1)ctx.moveTo(xx,yy);else ctx.lineTo(xx,yy);}
    ctx.stroke();
    if(times.length===1){ctx.fillStyle=colors[key];ctx.beginPath();ctx.arc(x(times[0]),y(value(kind,key,0)),2,0,2*Math.PI);ctx.fill();}
  }
  if(hover!==null){ctx.strokeStyle='#d7e8f5';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x(times[hover]),top);ctx.lineTo(x(times[hover]),bottom);ctx.stroke();}
  ctx.restore();
  $(kind+'-history-stats').textContent=series.map(key=>{const s=meta.statistics[kind][key];return `${key==='norm'?'模长':key.toUpperCase()}：均值 ${number(s.mean)} · 峰峰值 ${number(s.peak_to_peak)} · σ ${number(s.std)}`;}).join('\n');
}
function redraw(){draw('force');draw('moment');}
for(const [kind,canvas] of Object.entries(canvases)){
  new ResizeObserver(()=>draw(kind)).observe(canvas);
  canvas.addEventListener('pointermove',event=>{
    if(!history)return;
    const x=event.clientX-canvas.getBoundingClientRect().left;
    const times=history.times,target=times[0]+Math.max(0,Math.min(1,(x-74)/(canvas.clientWidth-92)))*(times[times.length-1]-times[0]);
    let lo=0,hi=times.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<target)lo=mid+1;else hi=mid;}
    if(lo>0&&target-times[lo-1]<times[lo]-target)lo--;
    hover=lo;
    $('history-cursor').textContent=`t = ${times[lo].toFixed(6)} s · 第 ${history.steps[lo]} 步 · Δt ${(history.dt[lo]*1e6).toFixed(2)} μs · `+['force','moment'].map(k=>`${k==='force'?'F':'M'} ${keys().map(key=>`${key==='norm'?'模长':key.toUpperCase()} ${number(value(k,key,lo))}`).join(' / ')} ${k==='force'?'N':'N·m'}`).join(' · ');
    redraw();
  });
  canvas.addEventListener('pointerleave',()=>{hover=null;$('history-cursor').textContent='指向曲线可查看每一个仿真步；虚线标记工况变化。';redraw();});
}
$('history-components').onchange=redraw;$('history-demean').onchange=redraw;
$('history-window').onchange=()=>{etag='';clearTimeout(timer);controller?.abort();fetchHistory();};
async function fetchHistory(){
  const request=new AbortController();controller=request;
  try{
    const seconds=$('history-window').value;
    const response=await fetch(`/api/loads?seconds=${seconds}`,{signal:request.signal,headers:etag?{'If-None-Match':etag}:{}});
    if(response.status===304)return;
    if(response.status===404){$('history-status').textContent='等待后端逐步采样，历史数据从新版后端启动时开始。';return;}
    if(!response.ok)throw new Error(`载荷历史 HTTP ${response.status}`);
    const buffer=await response.arrayBuffer(),view=new DataView(buffer);
    if(buffer.byteLength<4)throw new Error('载荷历史不完整');
    const length=view.getUint32(0,true);
    if(length>16*1024*1024||length+4>buffer.byteLength)throw new Error('载荷历史头无效');
    const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,length)));
    if(meta.version!==1||meta.stride!==44||meta.count<1||meta.count>65536||buffer.byteLength!==4+length+44*meta.count)throw new Error('载荷历史格式无效');
    const times=new Float64Array(meta.count),steps=new Float64Array(meta.count),dt=new Float32Array(meta.count),values=new Float32Array(meta.count*6);
    for(let i=0;i<meta.count;i++){
      const at=4+length+44*i;steps[i]=view.getUint32(at,true)+view.getUint32(at+4,true)*4294967296;
      times[i]=view.getFloat64(at+8,true);dt[i]=view.getFloat32(at+16,true);
      for(let k=0;k<6;k++)values[i*6+k]=view.getFloat32(at+20+4*k,true);
    }
    // Replace the entire server-owned window. Never append points from display packets.
    history={meta,times,steps,dt,values};hover=null;etag=response.headers.get('ETag');
    $('history-status').textContent=`逐仿真步采样 · ${meta.count.toLocaleString()} 点 · t = ${meta.start.toFixed(6)}–${meta.end.toFixed(6)} s · Δt ${(meta.dt_min*1e6).toFixed(1)}–${(meta.dt_max*1e6).toFixed(1)} μs${meta.capacity_limited?' · 已达缓冲容量上限':''}${meta.contiguous?'':' · 警告：步号存在缺口'}`;
    window.wind3dLoadHistory={meta,receivedAt:Date.now()};redraw();
  }catch(error){if(error.name!=='AbortError')$('history-status').textContent=`载荷历史请求失败，正在重试：${error.message}`;}
  finally{if(controller===request)timer=setTimeout(fetchHistory,2000);}
}
redraw();fetchHistory();
