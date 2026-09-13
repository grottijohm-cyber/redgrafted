// Exercise the actual page script with a small DOM and mocked RunPod responses.
// No API credentials, paid jobs, browser download, or npm dependencies required.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../client/client.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const savedKey = 'runpod-image-video-job-v1';
const success = {status:'COMPLETED', output:{status:'success', videos:[{
  type:'base64', data:Buffer.from('test-video').toString('base64'), filename:'output.mp4'
}]}};
const response = (body, status=200) => ({ok:status<400, status, json:async()=>body});
const flush = async()=>{for(let i=0;i<5;i++)await new Promise(setImmediate);};

function page(fetchHandler, saved=null) {
  const elements = new Map(), storage = new Map(), timers = new Map(), readers = [], calls=[];
  let timerId=0, blobId=0;
  function element(id) {
    if(!elements.has(id)) elements.set(id, {
      value:'', files:[], hidden:false, disabled:false, textContent:'', className:'', children:[], listeners:{},
      addEventListener(name,callback){this.listeners[name]=callback;},
      append(...items){this.children.push(...items);},
      replaceChildren(...items){this.children=items;}
    });
    return elements.get(id);
  }
  if(saved)storage.set(savedKey,JSON.stringify(saved));
  class BrowserURL extends URL {
    static createObjectURL(){return 'blob:test-'+(++blobId);}
    static revokeObjectURL(){}
  }
  class Reader {
    readAsDataURL(){readers.push(this);}
  }
  const context=vm.createContext({
    document:{getElementById:element, createElement:tag=>({...element(Symbol(tag)),tagName:tag})},
    localStorage:{getItem:key=>storage.get(key)||null, setItem:(key,value)=>storage.set(key,value), removeItem:key=>storage.delete(key)},
    URL:BrowserURL, Blob, Uint8Array, AbortController, TypeError, FileReader:Reader,
    atob:text=>Buffer.from(text,'base64').toString('binary'), confirm:()=>true,
    setTimeout:(callback,ms)=>{timers.set(++timerId,{callback,ms});return timerId;},
    clearTimeout:id=>timers.delete(id),
    fetch:async(url,options)=>{calls.push({url,options});return fetchHandler(url,options);}
  });
  vm.runInContext(source,context,{filename:'client.html'});
  element('endpoint').value='endpoint-test';element('key').value='test-only-api-key';
  element('imageUrl').value='https://example.com/mountain.png';
  element('prompt').value='Clouds drift slowly over the mountain.';
  return {element,storage,calls,readers,
    emit:(id,event)=>element(id).listeners[event]({preventDefault(){}}),
    tick:async()=>{for(const [id,timer] of [...timers])if(timer.ms===5000){timers.delete(id);timer.callback();}await flush();}
  };
}

test('image link + prompt submits one job and produces a download without saving the API key',async()=>{
  const p=page(async url=>response(url.endsWith('/run')?{id:'job-1'}:success));
  await p.emit('form','submit');await flush();
  assert.equal(p.calls.length,2);
  const submitted=JSON.parse(p.calls[0].options.body);
  assert.deepEqual(submitted.input,{image:'https://example.com/mountain.png',prompt:'Clouds drift slowly over the mountain.'});
  assert.equal(p.calls[0].options.headers.Authorization,'Bearer test-only-api-key');
  assert.match(p.element('status').textContent,/Video ready/);
  assert.equal(p.element('results').children[1].download,'output.mp4');
  assert.equal(p.element('generate').disabled,false);
  assert.equal(p.storage.size,0);
});

test('repeated submit while reading an image cannot create duplicate jobs',async()=>{
  const p=page(async url=>response(url.endsWith('/run')?{id:'job-1'}:success));
  p.element('image').files=[{size:100}];p.element('imageUrl').value='';
  const first=p.emit('form','submit');
  await p.emit('form','submit');
  assert.equal(p.readers.length,1);
  p.readers[0].result='data:image/png;base64,dGVzdA==';p.readers[0].onload();
  await first;await flush();
  assert.equal(p.calls.filter(call=>call.url.endsWith('/run')).length,1);
});

test('a failed status check reconnects to the saved job without resubmitting',async()=>{
  let statusChecks=0;
  const p=page(async url=>{
    if(url.endsWith('/run'))return response({id:'job-1'});
    if(++statusChecks===1)return response({},429);
    return response(success);
  });
  await p.emit('form','submit');await flush();
  assert.equal(p.element('resume').hidden,false);
  assert.equal(p.element('generate').disabled,true);
  assert.deepEqual(JSON.parse(p.storage.get(savedKey)),{endpoint:'endpoint-test',jobId:'job-1'});
  await p.emit('resume','click');await flush();
  assert.equal(p.calls.filter(call=>call.url.endsWith('/run')).length,1);
  assert.match(p.element('status').textContent,/Video ready/);
});

test('an error from a cancelled old poll cannot overwrite the next job',async()=>{
  let submitted=0, rejectOld;
  const p=page(async url=>{
    if(url.endsWith('/run'))return response({id:'job-'+(++submitted)});
    if(url.includes('/cancel/'))return response({status:'CANCELLED'});
    if(url.endsWith('/status/job-1'))return new Promise((resolve,reject)=>{rejectOld=reject;});
    return response({status:'IN_PROGRESS',output:{stage:'New generation running'}});
  });
  await p.emit('form','submit');await flush();
  await p.emit('cancel','click');
  await p.emit('form','submit');await flush();
  rejectOld(Error('Old request failed'));await flush();
  assert.equal(p.element('status').textContent,'New generation running');
  assert.equal(JSON.parse(p.storage.get(savedKey)).jobId,'job-2');
});

test('a delayed cancellation response cannot clear a newer job',async()=>{
  let submitted=0, completeOld, finishCancel;
  const p=page(async url=>{
    if(url.endsWith('/run'))return response({id:'job-'+(++submitted)});
    if(url.includes('/cancel/'))return new Promise(resolve=>{finishCancel=resolve;});
    if(url.endsWith('/status/job-1'))return new Promise(resolve=>{completeOld=resolve;});
    return response({status:'IN_PROGRESS',output:{stage:'New generation running'}});
  });
  await p.emit('form','submit');await flush();
  const cancel=p.emit('cancel','click');
  completeOld(response(success));await flush();
  await p.emit('form','submit');await flush();
  finishCancel(response({status:'CANCELLED'}));await cancel;await flush();
  assert.equal(p.element('status').textContent,'New generation running');
  assert.equal(JSON.parse(p.storage.get(savedKey)).jobId,'job-2');
});

test('a worker error at completion releases controls and clears the saved job',async()=>{
  const p=page(async url=>response(url.endsWith('/run')?{id:'job-1'}:{status:'COMPLETED',output:{error:'Model preparation failed'}}));
  await p.emit('form','submit');await flush();
  assert.match(p.element('status').textContent,/Model preparation failed/);
  assert.equal(p.element('generate').disabled,false);
  assert.equal(p.storage.size,0);
});

test('a reopened page resumes an existing job without another run request',async()=>{
  const p=page(async()=>response(success),{endpoint:'endpoint-test',jobId:'saved-job'});
  await p.emit('resume','click');await flush();
  assert.equal(p.calls.length,1);
  assert.match(p.calls[0].url,/\/status\/saved-job$/);
  assert.match(p.element('status').textContent,/Video ready/);
});
