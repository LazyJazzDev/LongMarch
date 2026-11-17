import * as THREE from 'three';

// Same lighting and linear colors as display.slang.
export function aircraftMaterial(){return new THREE.ShaderMaterial({vertexColors:true,side:THREE.DoubleSide,
  vertexShader:`varying vec3 n,c;void main(){n=normal;c=color;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader:`varying vec3 n,c;void main(){float l=.28+.72*abs(dot(normalize(n),normalize(vec3(-.5,1.,.3))));gl_FragColor=vec4(c*l,1.);}`});}

export function streamMesh(){
  const material=new THREE.ShaderMaterial({transparent:true,depthWrite:false,side:THREE.DoubleSide,
    uniforms:{viewport:{value:new THREE.Vector2(1,1)},width:{value:1.5},lower:{value:.01},upper:{value:.15},opaqueDepth:{value:null},useDepth:{value:0}},
    vertexShader:`attribute vec3 start,end,previous,next,tint;attribute vec2 history;
      uniform vec2 viewport;uniform float width;varying vec2 local;varying float len,k;varying vec3 color;varying vec2 caps;
      void main(){
        vec4 a=projectionMatrix*modelViewMatrix*vec4(start,1.),b=projectionMatrix*modelViewMatrix*vec4(end,1.);
        caps=vec2(all(equal(previous,start))?1.:0.,all(equal(next,end))?1.:0.);
        if(a.z < -a.w && b.z < -b.w){gl_Position=vec4(0,0,2,1);return;}
        float da=a.z+a.w,db=b.z+b.w;
        if(da<0.){a=mix(a,b,da/(da-db));caps.x=1.;}
        if(db<0.){b=mix(b,a,db/(db-(a.z+a.w)));caps.y=1.;}
        vec2 delta=(b.xy/b.w-a.xy/a.w)*viewport*.5;len=length(delta);
        vec2 direction=delta/max(len,.0001),normal=vec2(-direction.y,direction.x);
        float side=position.y,endpoint=position.x,extent=width*.5+1.;
        vec4 clip=endpoint>0.?b:a;float cap=endpoint>0.?caps.y:caps.x;vec2 join=normal;
        if(cap==0.){vec4 neighbor=projectionMatrix*modelViewMatrix*vec4(endpoint>0.?next:previous,1.);
          if(neighbor.z > -neighbor.w){vec2 adj=(neighbor.xy/neighbor.w-clip.xy/clip.w)*viewport*(endpoint>0.?.5:-.5);
            if(length(adj)>.0001){adj=normalize(adj);vec2 m=normal+vec2(-adj.y,adj.x);
              if(length(m)>.0001){m=normalize(m);join=m/max(dot(m,normal),.5);}}}}
        float extension=(endpoint*2.-1.)*extent*cap;
        clip.xy+=(direction*extension+join*side*extent)*2./viewport*clip.w;
        gl_Position=clip;local=vec2(endpoint*len+extension,side*extent);k=mix(history.x,history.y,endpoint);color=tint;
      }`,
    fragmentShader:`uniform float width,lower,upper,useDepth;uniform vec2 viewport;uniform sampler2D opaqueDepth;varying vec2 local,caps;varying float len,k;varying vec3 color;
      void main(){if(useDepth>.5&&gl_FragCoord.z>texture2D(opaqueDepth,gl_FragCoord.xy/viewport).r+.00001)discard;
        float cap=max(max(-local.x*caps.x,(local.x-len)*caps.y),0.);
        float coverage=clamp(.5-(length(vec2(cap,local.y))-width*.5),0.,1.);
        float emphasis=clamp((k-lower)/(upper-lower),0.,1.);
        if(coverage*emphasis<=0.)discard;
        gl_FragColor=vec4(mix(color,vec3(1.,.35,.06),emphasis),coverage*emphasis);}`});
  const mesh=new THREE.Mesh(new THREE.InstancedBufferGeometry(),material);mesh.frustumCulled=false;mesh.renderOrder=5;return mesh;
}

export function setStreams(mesh,vertices){
  const count=vertices.length/18,g=new THREE.InstancedBufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute([0,-1,0,1,-1,0,0,1,0,0,1,0,1,-1,0,1,1,0],3));
  const attributes=Object.fromEntries(['start','end','previous','next','tint','history'].map(name=>[name,new Float32Array(count*(name==='history'?2:3))]));
  const equal=(a,b)=>vertices[a]===vertices[b]&&vertices[a+1]===vertices[b+1]&&vertices[a+2]===vertices[b+2];
  for(let i=0;i<vertices.length;i+=18){
    const segment=i/18,at=segment*3;
    const previous=i>=18&&equal(i-9,i)?i-18:i,next=i+18<vertices.length&&equal(i+18,i+9)?i+27:i+9;
    for(let j=0;j<3;j++){
      attributes.start[at+j]=vertices[i+j];attributes.end[at+j]=vertices[i+9+j];
      attributes.previous[at+j]=vertices[previous+j];attributes.next[at+j]=vertices[next+j];attributes.tint[at+j]=vertices[i+6+j];
    }
    attributes.history[segment*2]=vertices[i+3];attributes.history[segment*2+1]=vertices[i+12];
  }
  for(const [name,values] of Object.entries(attributes))g.setAttribute(name,new THREE.InstancedBufferAttribute(values,name==='history'?2:3));
  g.instanceCount=count;mesh.geometry.dispose();mesh.geometry=g;
}

export function volumeRenderer(renderer,scene,camera,streams){
  const target=new THREE.WebGLRenderTarget(1,1,{depthTexture:new THREE.DepthTexture(1,1)});
  const output=new THREE.WebGLRenderTarget(1,1);
  const uniforms={background:{value:target.texture},depth:{value:target.depthTexture},field:{value:null},
    inverse:{value:new THREE.Matrix4()},eye:{value:new THREE.Vector3()},density:{value:.035},threshold:{value:6},
    stepSize:{value:.0625},colored:{value:0},enabled:{value:0}};
  const material=new THREE.ShaderMaterial({glslVersion:THREE.GLSL3,depthTest:false,depthWrite:false,uniforms,
    vertexShader:`out vec2 uv0;void main(){uv0=position.xy*.5+.5;gl_Position=vec4(position.xy,0,1);}`,
    fragmentShader:`precision highp sampler3D;uniform sampler2D background,depth;uniform sampler3D field;
      uniform mat4 inverse;uniform vec3 eye;uniform float density,threshold,stepSize,colored,enabled;in vec2 uv0;out vec4 result;
      float sampleField(vec3 p){return texture(field,(p-vec3(-6,-4,-6))/vec3(16,8,12)).r;}
      vec3 palette(float v){float t=clamp(log2(1.+v)/7.,0.,1.);return mix(mix(vec3(.04,.24,.43),vec3(.08,.76,.73),clamp(t*2.,0.,1.)),vec3(1.,.68,.25),clamp((t-.5)*2.,0.,1.));}
      void main(){vec3 bg=texture(background,uv0).rgb;if(enabled<.5){result=vec4(bg,1);return;}
        vec2 ndc=uv0*2.-1.;vec4 farP=inverse*vec4(ndc,1,1);vec3 dir=normalize(farP.xyz/farP.w-eye);
        vec3 a=(vec3(-6,-4,-6)-eye)/(dir+1e-8),b=(vec3(10,4,6)-eye)/(dir+1e-8);
        vec3 entry=min(a,b),exitP=max(a,b);float first=max(0.,max(entry.x,max(entry.y,entry.z))),last=min(exitP.x,min(exitP.y,exitP.z));
        float d=texture(depth,uv0).r;if(d<1.){vec4 surface=inverse*vec4(ndc,d*2.-1.,1);last=min(last,length(surface.xyz/surface.w-eye));}
        vec3 color=vec3(0);float trans=1.;
        for(float t=first+.5*stepSize;t<last&&trans>.012;t+=stepSize){vec3 p=eye+dir*t;float value=sampleField(p),rho=max(0.,value-threshold)*density;if(rho<=0.)continue;
          float alpha=1.-exp(-rho*stepSize);vec3 tint=colored<.5?vec3(.75,.81,.85):palette(value);float shadow=0.;
          for(int j=1;j<=3;j++){vec3 q=p+vec3(-.16,.24,.12)*float(j);if(all(greaterThan(q,vec3(-6,-4,-6)))&&all(lessThan(q,vec3(10,4,6))))shadow+=max(0.,sampleField(q)-threshold);}
          tint*=.25+.75*exp(-shadow*density*.24);color+=trans*alpha*tint;trans*=1.-alpha;
        }result=vec4(color+trans*bg,1.);}`});
  const quadScene=new THREE.Scene();quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),material));
  const presentScene=new THREE.Scene();
  presentScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.ShaderMaterial({depthTest:false,depthWrite:false,
    uniforms:{image:{value:output.texture}},vertexShader:'varying vec2 uv0;void main(){uv0=position.xy*.5+.5;gl_Position=vec4(position.xy,0,1);}',
    fragmentShader:`uniform sampler2D image;varying vec2 uv0;void main(){gl_FragColor=texture2D(image,uv0);
      #include <colorspace_fragment>
    }`})));
  let texture=null;
  return {
    set(data,n){texture?.dispose();texture=new THREE.Data3DTexture(data,n,n/2,n*3/4);texture.format=THREE.RedFormat;texture.type=THREE.FloatType;texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.unpackAlignment=1;texture.needsUpdate=true;uniforms.field.value=texture;},
    render(view){
      const size=renderer.getDrawingBufferSize(new THREE.Vector2());if(target.width!==size.x||target.height!==size.y){target.setSize(size.x,size.y);output.setSize(size.x,size.y);}
      streams.material.uniforms.viewport.value.copy(size);
      streams.material.uniforms.useDepth.value=0;
      uniforms.inverse.value.multiplyMatrices(camera.matrixWorld,camera.projectionMatrixInverse);uniforms.eye.value.copy(camera.position);
      uniforms.density.value=view?.volume_density??.035;uniforms.threshold.value=view?.volume_threshold??6;uniforms.colored.value=view?.colored_volume??0;
      uniforms.stepSize.value=(view?.volume_step??1)*16/(texture?.image.width??64);uniforms.enabled.value=texture&&view?.volume?1:0;
      const visible=streams.visible;streams.visible=false;renderer.setRenderTarget(target);renderer.render(scene,camera);
      renderer.setRenderTarget(output);renderer.render(quadScene,camera);
      // Depth from the opaque pass is reused by the streamline shader through its own scene pass.
      streams.visible=visible;
      streams.material.uniforms.opaqueDepth.value=target.depthTexture;streams.material.uniforms.useDepth.value=1;
      if(visible){const oldAuto=renderer.autoClear;renderer.autoClear=false;
        const hidden=[];scene.traverse(o=>{if(o!==streams&&(o.isMesh||o.isLine)&&o.visible){hidden.push(o);o.visible=false;}});
        renderer.render(scene,camera);for(const o of hidden)o.visible=true;renderer.autoClear=oldAuto;}
      renderer.setRenderTarget(null);renderer.render(presentScene,camera);
    }
  };
}
