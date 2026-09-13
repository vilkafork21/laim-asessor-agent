"""Read-only fallback for this audit when PyArrow cannot be installed.
Supports compact-Thrift metadata, PLAIN/dictionary pages, RLE levels, and
uncompressed/Snappy/gzip/Zstd columns. Unknown encodings fail explicitly.
Nested leaves are returned separately, NOT reconstructed as original objects.
For production, use PyArrow and independently verify audit outputs.
Format: https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift
"""
from pathlib import Path
import struct, ctypes, ctypes.util, gzip
from thrift.Thrift import TType
from thrift.transport.TTransport import TMemoryBuffer
from thrift.protocol.TCompactProtocol import TCompactProtocol

def _value(p,t):
    if t==TType.STRUCT: return _struct(p)
    if t in (TType.LIST,TType.SET):
        et,n = p.readListBegin(); a=[_value(p,et) for _ in range(n)]; p.readListEnd(); return a
    if t==TType.MAP:
        kt,vt,n=p.readMapBegin(); a={_value(p,kt):_value(p,vt) for _ in range(n)};p.readMapEnd();return a
    fun={TType.BOOL:p.readBool,TType.BYTE:p.readByte,TType.I16:p.readI16,TType.I32:p.readI32,TType.I64:p.readI64,TType.DOUBLE:p.readDouble,TType.STRING:p.readBinary}
    return fun[t]()
def _struct(p):
    out={};p.readStructBegin()
    while True:
        _,t,i=p.readFieldBegin()
        if t==TType.STOP: break
        out[i]=_value(p,t);p.readFieldEnd()
    p.readStructEnd();return out

def thrift(data):
    t=TMemoryBuffer(data);p=TCompactProtocol(t);s=_struct(p);return s,t.cstringio_buf.tell()
def text(v):return v.decode('utf8') if isinstance(v,bytes) else v

def metadata(path):
    with open(path,'rb') as f:
        assert f.read(4)==b'PAR1';f.seek(-8,2);tail=f.read(8);assert tail[4:]==b'PAR1'
        n=struct.unpack('<I',tail[:4])[0];f.seek(-8-n,2);return thrift(f.read(n))[0]
def leaves(meta):
    out={};els=iter(meta[2]);root=next(els)
    def visit(path,definition,repetition):
        e=next(els);path=path+[text(e[4])];rt=e.get(3,0)
        d=definition+int(rt in (1,2));r=repetition+int(rt==2)
        for _ in range(e.get(5,0)):visit(path,d,r)
        if 1 in e:out[tuple(path)]={'schema':e,'max_def':d,'max_rep':r}
    for _ in range(root.get(5,0)):visit([],0,0)
    return out

def _uvar(data,pos):
    v=shift=0
    while True:
        b=data[pos];pos+=1;v|=(b&127)<<shift
        if b<128:return v,pos
        shift+=7
        if shift>70:raise ValueError('oversize varint')
def hybrid(data,width,n):
    if n==0:return []
    if width==0:return [0]*n
    out=[];pos=0
    while len(out)<n:
        h,pos=_uvar(data,pos)
        if h&1:
            count=(h>>1)*8;nb=count*width//8
            raw=int.from_bytes(data[pos:pos+nb],'little');pos+=nb;mask=(1<<width)-1
            out.extend((raw>>(j*width))&mask for j in range(count))
        else:
            count=h>>1;nb=(width+7)//8;v=int.from_bytes(data[pos:pos+nb],'little');pos+=nb;out.extend([v]*count)
    return out[:n]
def _plain(data,typ,n,flen=None):
    if typ==0:return [bool((data[i//8]>>(i%8))&1) for i in range(n)]
    if typ in (1,2,4,5):
        fmt={1:'i',2:'q',4:'f',5:'d'}[typ];sz=struct.calcsize(fmt);return list(struct.unpack('<'+fmt*n,data[:n*sz]))
    if typ==6:
        out=[];pos=0
        for _ in range(n):
            sz=struct.unpack_from('<I',data,pos)[0];pos+=4;v=data[pos:pos+sz];pos+=sz
            try:v=v.decode('utf8')
            except UnicodeDecodeError:pass
            out.append(v)
        return out
    if typ in (3,7):
        sz=12 if typ==3 else flen;return [data[i*sz:(i+1)*sz] for i in range(n)]
    raise NotImplementedError(('plain',typ))
_snappy=None;_zstd=None

def decompress(data,codec,size):
    global _snappy,_zstd
    if codec==0:return data
    if codec==2:return gzip.decompress(data)
    if codec==1:
        if _snappy is None:
            _snappy=ctypes.CDLL(ctypes.util.find_library('snappy'))
            _snappy.snappy_uncompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]
            _snappy.snappy_uncompress.restype=ctypes.c_int
        dest=ctypes.create_string_buffer(size);sz=ctypes.c_size_t(size)
        ret=_snappy.snappy_uncompress(data,len(data),dest,ctypes.byref(sz))
        if ret:raise ValueError(('snappy',ret))
        assert sz.value==size;return dest.raw[:sz.value]
    if codec==6:
        if _zstd is None:
            _zstd=ctypes.CDLL(ctypes.util.find_library('zstd'))
            _zstd.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t];_zstd.ZSTD_decompress.restype=ctypes.c_size_t
        dest=ctypes.create_string_buffer(size);sz=_zstd.ZSTD_decompress(dest,size,data,len(data))
        assert sz==size;return dest.raw[:sz]
    raise NotImplementedError(('codec',codec))

def _levels_v1(data,pos,maxlev,n):
    if not maxlev:return [0]*n,pos
    size=struct.unpack_from('<I',data,pos)[0];pos+=4
    return hybrid(data[pos:pos+size],maxlev.bit_length(),n),pos+size

def read_leaf(path,leaf,meta=None):
    meta=meta or metadata(path); info=leaves(meta)[tuple(leaf)];all_vals=[];all_rep=[];all_def=[]
    for rg in meta[4]:
        matches=[c[3] for c in rg[1] if tuple(map(text,c[3][3]))==tuple(leaf)]
        if len(matches)!=1:raise ValueError('missing or repeated column')
        cm=matches[0];start=min(cm[9],cm.get(11,cm[9]));dictionary=None;nv=0
        with open(path,'rb') as f:f.seek(start);blob=f.read(cm[7])
        pos=0
        while nv<cm[5]:
            h,used=thrift(blob[pos:]);pos+=used;body=blob[pos:pos+h[3]];pos+=h[3]
            if h[1]==2:
                raw=decompress(body,cm[4],h[2]);dh=h[7];assert dh[2] in (0,2)
                dictionary=_plain(raw,cm[1],dh[1],info['schema'].get(2));continue
            if h[1] not in (0,3):continue
            if h[1]==0:
                dh=h[5];n=dh[1];enc=dh[2];raw=decompress(body,cm[4],h[2]);p=0
                rep,p=_levels_v1(raw,p,info['max_rep'],n);defs,p=_levels_v1(raw,p,info['max_def'],n);raw=raw[p:]
            else:
                dh=h[8];n=dh[1];enc=dh[4];nr=dh[6];nd=dh[5]
                rep=hybrid(body[:nr],info['max_rep'].bit_length(),n) if info['max_rep'] else [0]*n
                defs=hybrid(body[nr:nr+nd],info['max_def'].bit_length(),n) if info['max_def'] else [0]*n
                raw=decompress(body[nr+nd:],cm[4],h[2]-nr-nd) if dh.get(7,True) else body[nr+nd:]
            nn=sum(d==info['max_def'] for d in defs)
            if nn==0:values=[]
            elif enc==0:values=_plain(raw,cm[1],nn,info['schema'].get(2))
            elif enc in (2,8):
                if dictionary is None:raise ValueError('no dictionary')
                values=[dictionary[i] for i in hybrid(raw[1:],raw[0],nn)]
            elif enc==3 and cm[1]==0:values=hybrid(raw[4:],1,nn)
            else:raise NotImplementedError(('encoding',enc,'column',leaf))
            vals=iter(values);all_vals.extend(next(vals) if d==info['max_def'] else None for d in defs)
            all_rep.extend(rep);all_def.extend(defs);nv+=n
        assert nv==cm[5]
    if not info['max_rep']:
        assert len(all_vals)==meta[3],(len(all_vals),meta[3]);return all_vals
    rows=[]
    for rep,d,v in zip(all_rep,all_def,all_vals):
        if rep==0:rows.append([])
        if v is not None:rows[-1].append(v)
    assert len(rows)==meta[3],(len(rows),meta[3]);return rows

def read_columns(path,names=None):
    m=metadata(path); ls=leaves(m);out={}
    for leaf in ls:
        key='.'.join(leaf)
        if names is None or key in names:out[key]=read_leaf(path,leaf,m)
    return out
