import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const DropZoneContainer = styled.div`
  width: 400px;
  height: 200px;
  border: 2px dashed #509AD1;
  border-radius: 10px;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  margin: 20px;
`;

const UploadMessage = styled.p`
  font-size: 16px;
  color: #509AD1;
`;

function PhotoUploader() {
  const onDrop = useCallback((acceptedFiles) => {
    console.log(acceptedFiles); // 업로드된 파일 리스트
    // 파일을 서버로 전송하거나, 로컬에 미리보기로 표시할 수 있습니다.
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: 'image/*' });

  return (
    <DropZoneContainer {...getRootProps()}>
      <input {...getInputProps()} />
      {isDragActive ? (
        <UploadMessage>Drop the files here...</UploadMessage> // 드래그 해서 올려놓았을 때
      ) : (
        <UploadMessage>Drag & drop some images here, or click to select files</UploadMessage> // 드래그 하기 전
      )}
    </DropZoneContainer>
  );
}

export default PhotoUploader;
