import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const $=id=>document.getElementById(id);
const query=new URLSearchParams(location.search);
document.querySelector('aside').style.display='none';
$('load-history').style.display='none';
document.querySelector('main').style.gridTemplateColumns='1fr';
document.querySelector('.legend').style.display='none';
document.querySelector('.scale-overlay').style.display='none';
document.querySelector('.eyebrow').textContent='AIRCRAFT BODY COORDINATES';
document.querySelector('h1').textContent='以几何参考点为原点';
document.title='Wind3D · 飞机参考原点';
$('connection').textContent='几何预览';
$('hardware').textContent='参考点不是由质量分布计算的重心';
document.querySelector('.view-foot').textContent='拖拽旋转 · 滚轮缩放 · 右键平移 · 内部坐标标记透视显示';
document.querySelector('.view-tools').innerHTML='<button id="origin-side">侧视</button><button id="origin-top">俯视</button><button id="origin-oblique">斜视</button><button id="origin-close">原点特写</button><a href="/">返回风洞</a>';

const style=document.createElement('style');
style.textContent='.origin-label{position:absolute;pointer-events:none;z-index:3;white-space:nowrap;font-size:13px;padding:4px 7px;border-radius:4px;background:#101923de;color:#f3f5f7}.origin-note{position:absolute;right:20px;top:100px;max-width:330px;z-index:3;background:#101923e8;border:1px solid #344453;padding:15px;border-radius:8px;font-size:13px;line-height:1.8}.origin-note strong{color:#fff}.origin-note p{margin:6px 0}@media(max-width:800px){.origin-note{top:155px;right:10px;max-width:260px;font-size:12px}}';
document.head.append(style);

async function show(){
  const snapshot=query.get('mesh');
  let angle=0,reference;
  let meshID=snapshot;
  if(snapshot){
    if(!/^\d+$/.test(snapshot))throw new Error('无效的模型编号');
    const x=Number(query.get('reference-x'));
    if(!query.has('reference-x')||!Number.isFinite(x)||Math.abs(x)>10)throw new Error('需要有效的参考点坐标');
    reference=new THREE.Vector3(x,0,0);
  }else{
    const response=await fetch('/api/frame');
    if(!response.ok)throw new Error('暂未提供模型快照');
    const bytes=await response.arrayBuffer(),length=new DataView(bytes).getUint32(0,true);
    if(length>65536||length+4>bytes.byteLength)throw new Error('快照头无效');
    const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,4,length)));
    if(!meta.reference_cg_body?.every(Number.isFinite)||meta.reference_cg_body.length!==3)throw new Error('快照未提供几何参考点');
    reference=new THREE.Vector3(...meta.reference_cg_body);
    angle=meta.controls.angle*Math.PI/180;
    meshID=meta.mesh;
  }
  const response=await fetch('/api/mesh/'+meshID);
  if(!response.ok)throw new Error('模型网格读取失败');
  const bytes=await response.arrayBuffer();
  if(!bytes.byteLength||bytes.byteLength%36)throw new Error('模型网格长度无效');
  const geometry=new THREE.BufferGeometry(),buffer=new THREE.InterleavedBuffer(new Float32Array(bytes),9);
  geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));
  geometry.setAttribute('normal',new THREE.InterleavedBufferAttribute(buffer,3,3));
  geometry.setAttribute('color',new THREE.InterleavedBufferAttribute(buffer,3,6));
  const renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(43,1,.01,100);
  const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.minDistance=.6;orbit.maxDistance=30;
  scene.add(new THREE.HemisphereLight(0xe5edf4,0x52565c,2));
  const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(-3,8,6);scene.add(light);
  const aircraft=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.72,metalness:.08,side:THREE.DoubleSide}));
  // Source meshes are in world axes. Undo aircraft pitch, then translate in body axes.
  aircraft.rotation.z=angle;aircraft.position.copy(reference).negate();scene.add(aircraft);aircraft.updateMatrixWorld(true);
  const grid=new THREE.GridHelper(10,10,0x516273,0x263542);grid.position.y=-.7;scene.add(grid);
  const axes=new THREE.AxesHelper(1.15);axes.material.depthTest=false;axes.material.depthWrite=false;axes.renderOrder=10;scene.add(axes);
  const oldOrigin=reference.clone().negate(),newOrigin=new THREE.Vector3();
  function dot(point,color){const mesh=new THREE.Mesh(new THREE.SphereGeometry(.025,16,12),new THREE.MeshBasicMaterial({color,depthTest:false,depthWrite:false}));mesh.position.copy(point);mesh.renderOrder=12;scene.add(mesh);}
  dot(newOrigin,0xffffff);dot(oldOrigin,0xffc66b);
  const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([newOrigin,oldOrigin]),new THREE.LineBasicMaterial({color:0xffc66b,depthTest:false,depthWrite:false}));line.renderOrder=11;scene.add(line);
  const labels=[];
  function label(text,point,dx,dy,color){const el=document.createElement('span');el.className='origin-label';el.textContent=text;el.style.color=color;$('viewport').append(el);labels.push({el,point,dx,dy});}
  label('新原点 · 几何参考点',newOrigin,-150,-42,'#fff');
  label('原机身原点',oldOrigin,12,20,'#ffc66b');
  label('+X 下游 / 机尾',new THREE.Vector3(1.15,0,0),8,-12,'#ff8989');
  label('+Y 上方',new THREE.Vector3(0,1.15,0),8,-12,'#89ef9e');
  label('+Z 右舷',new THREE.Vector3(0,0,1.15),8,-12,'#99bbff');
  const note=document.createElement('div');note.className='origin-note';
  const coordinates=v=>v.toArray().map(x=>Math.abs(x)<1e-9?'0':x.toFixed(6)).join(', ');
  note.innerHTML='<strong>同一架飞机，只换坐标原点</strong><p id="reference-coordinate"></p><p id="previous-coordinate"></p><p>坐标轴方向保持不变。质量分布、外形及舵面状态均未改变；本页不表示真实重心已确定。</p><p id="origin-extents"></p>';
  $('viewport').append(note);
  $('reference-coordinate').textContent=`新原点在旧坐标中：(${coordinates(reference)}) m`;
  $('previous-coordinate').textContent=`旧原点在新坐标中：(${coordinates(oldOrigin)}) m`;
  const box=new THREE.Box3().setFromObject(aircraft);
  $('origin-extents').textContent=`新坐标：机头 X≈${box.min.x.toFixed(3)} m，机尾 X≈${box.max.x.toFixed(3)} m。`;
  $('applied').textContent=`p′ = p − (${coordinates(reference)}) m · 机身坐标视图（已去除来流迎角）`;
  function view(name){camera.up.set(0,1,0);orbit.target.set(0,0,0);if(name==='top'){camera.up.set(0,0,-1);camera.position.set(0,9,0);}else if(name==='oblique')camera.position.set(-6,3.5,7);else if(name==='close')camera.position.set(0,.18,1.8);else camera.position.set(0,.25,8.5);orbit.update();}
  for(const name of ['side','top','oblique','close'])$('origin-'+name).onclick=()=>view(name);
  view('side');
  new ResizeObserver(()=>{const host=$('viewport');renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();}).observe($('viewport'));
  function render(){requestAnimationFrame(render);orbit.update();renderer.render(scene,camera);const width=renderer.domElement.clientWidth,height=renderer.domElement.clientHeight;for(const item of labels){const p=item.point.clone().project(camera);item.el.hidden=p.z< -1||p.z>1;item.el.style.left=((p.x+1)*width/2+item.dx)+'px';item.el.style.top=((1-p.y)*height/2+item.dy)+'px';}}
  render();
  window.wind3dOriginView={reference:reference.toArray(),oldOrigin:oldOrigin.toArray(),newOrigin:reference.clone().sub(reference).toArray(),bodyCoordinates:true,mesh:meshID,vertices:buffer.count,bounds:[box.min.toArray(),box.max.toArray()]};
}
show().catch(error=>{$('error').hidden=false;$('error').textContent=error.message;$('connection').textContent='模型加载失败';});
